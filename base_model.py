"""Shared model + tokenizer + checkpoint helpers for the whole pipeline.

A small GPT, coded from scratch. Decoder-only transformer. Used by
pretrain / sft / lora / dpo / eval. kv_cache.py carries its own copy of the
attention/forward path, extended with a KV cache (weights stay compatible).

Data flow: tokens -> token + position embeddings -> n_layer blocks of
[causal self-attention, MLP] with pre-norm residuals -> LayerNorm -> logits.
Causal masking (no token attends forward) is what makes it autoregressive.

Checkpoints saved with save_checkpoint() bundle the weights, the config and
the tokenizer vocab, so load_checkpoint() can rebuild a working
(model, tokenizer) pair from the single .pt file.
"""

import math
import os
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, "data")

# One checkpoint per pipeline stage; each stage loads the previous one.
CKPT_BASE = os.path.join(BASE_PATH, "ckpt_base.pt")  # pretrain.py
CKPT_SFT = os.path.join(BASE_PATH, "ckpt_sft.pt")    # sft.py (loads base)
CKPT_LORA = os.path.join(BASE_PATH, "ckpt_lora.pt")  # lora.py (loads base, adapters only)
CKPT_DPO = os.path.join(BASE_PATH, "ckpt_dpo.pt")    # dpo.py (loads sft)

"""# Tokenizer"""

EOT = "<|endoftext|>"
PAD = "[PAD]"
UNK = "[UNK]"
SPECIALS = (UNK, PAD, EOT)  # reserved tokens; take ids 0, 1, 2

# Splitting on this regex keeps special tokens intact as their own pieces,
# because the capture group makes re.split return the separators too.
SPECIALS_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")

# A token is either a run of word characters or a run of punctuation.
# Whitespace matches neither, so it simply disappears.
PRETOK = re.compile(r"\w+|[^\w\s]+")


# Split raw text into word/punctuation pieces: "hello world!" -> ["hello", "world", "!"]
def pretokenize(text):
    return PRETOK.findall(text)


class WordTokenizer:
    # vocab is an ordered list of tokens; a token's position is its id.
    def __init__(self, vocab):
        self.id_to_token = list(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}
        self.unk_id = self.token_to_id[UNK]
        self.pad_id = self.token_to_id[PAD]
        self.eot_id = self.token_to_id[EOT]
        self.specials = set(SPECIALS)

    # Text -> token ids. Special tokens map to their own single id;
    # everything else is pretokenized and looked up, unknown words -> [UNK].
    def encode(self, text):
        ids = []
        for piece in SPECIALS_RE.split(text):
            if piece in self.specials:
                ids.append(self.token_to_id[piece])
            else:
                ids.extend(self.token_to_id.get(t, self.unk_id) for t in pretokenize(piece))
        return ids

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    # Token ids -> text. Out-of-range ids are dropped; specials are dropped
    # too unless skip_special=False. Tokens are rejoined with single spaces,
    # so decode(encode(x)) recovers the words but not the exact spacing.
    def decode(self, ids, skip_special=True):
        tokens = [self.id_to_token[i] for i in ids if 0 <= i < len(self.id_to_token)]
        if skip_special:
            tokens = [t for t in tokens if t not in self.specials]
        return " ".join(tokens)


"""# GPT Model"""


# Everything that determines parameter shapes lives here, so a checkpoint is
# only loadable into a model built with the same config.
@dataclass
class GPTConfig:
    vocab_size: int = 10000
    block_size: int = 256   # max context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384       # must be divisible by n_head
    dropout: float = 0.1
    bias: bool = True       # bias in Linear/LayerNorm, like GPT-2


# Multi-head causal self-attention: the only layer where tokens exchange
# information, and only backward (token t sees tokens 0..t).
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        # Each position emits a query ("what am I looking for"), a key ("what
        # do I contain") and a value ("what do I pass along if attended to").
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # 384*384 + 384 = 147,840
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # 147,840
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # 147,840
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # 147,840 -> attention total 591,360 - mixes heads back
        self.dropout_p = cfg.dropout                   # on the attention weights (inside SDPA)
        self.resid_dropout = nn.Dropout(cfg.dropout)   # on the output
        # Unused since attention moved to F.scaled_dot_product_attention
        # (is_causal=True builds the lower-triangular mask internally), but kept
        # registered so older checkpoints with blocks.*.attn.mask still load.
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        )

    def forward(self, x):                          # x: (B, T, 384)
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # softmax(q k^T / sqrt(hd)) v with causal masking and attention-weight
        # dropout, fused into one kernel.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )                                                         # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))                 # one return value, no (k, v)


def gelu(x):
    # x * P(N(0,1) <= x): pass x in proportion to how "typically positive" it is.
    # Big positive x -> ~x (like identity), big negative -> ~0 (like ReLU), with
    # a smooth dip around 0 instead of ReLU's hard corner.
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Position-wise feed-forward: attention moves info between tokens, this
# processes each token independently. Expand 4x, nonlinearity, project back.
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up_proj = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)    # 384*1536 + 1536 = 591,360
        self.down_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # 1536*384 + 384  = 590,208
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # GELU = smooth ReLU; the standard transformer activation
        return self.dropout(self.down_proj(gelu(self.up_proj(x))))


# One transformer block. Pre-norm (LayerNorm before each sublayer, not after)
# keeps a clean identity path through the residuals, which is what makes
# stacking many blocks trainable: each block adds a small correction to the
# running representation instead of replacing it.
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)  # 384 + 384 = 768
        self.attn = CausalSelfAttention(cfg)                      # 591,360
        self.mlp_norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)   # 768
        self.mlp = MLP(cfg)                                       # 1,181,568 -> block total 1,774,464

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))   # residual add: keep x, add attention's correction
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.n_embd)     # 10000*384 = 3,840,000
        self.embed_positions = nn.Embedding(cfg.block_size, cfg.n_embd)  # 256*384 = 98,304
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]) # 6 * 1,774,464 = 10,646,784
        self.norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)              # 768
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False) # 3,840,000, tied -> +0
        self.embed_tokens.weight = self.lm_head.weight  # weight tying: same matrix both ways
        self._init_weights()

    # GPT-2 init: weights ~ N(0, 0.02), biases zero. Residual output projections
    # (o_proj, down_proj) are scaled down by sqrt(2*n_layer) because each block
    # adds two residual contributions and we want the stream's variance flat with depth.
    def _init_weights(self):
        residual_projs = ("o_proj", "down_proj")
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layer) if name.endswith(residual_projs) else 0.02
                nn.init.normal_(m.weight, mean=0.0, std=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weight counted once

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)          # always 0..T-1, no offset
        x = self.drop(self.embed_tokens(idx) + self.embed_positions(pos))
        for block in self.blocks:                          # plain loop, nothing collected
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # per-token -log p(true next token); target -1 marks prompt/padding
            # positions, excluded from the mean by ignore_index
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]      # whole context, cropped to the window
            logits, _ = self(idx_cond)                    # re-run EVERY token through all 6 blocks
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx


"""# Checkpoint helpers"""


# One file holds everything needed to resume: weights, the config that fixes
# the parameter shapes, and the tokenizer vocab (so token ids keep meaning the
# same words across stages). `extra` is for stage metadata (step, reward mode...).
def save_checkpoint(path, model, tok, **extra):
    torch.save({
        "model": model.state_dict(),
        "config": model.cfg.__dict__,
        "vocab": tok.id_to_token,
        **extra,
    }, path)


# Rebuild (model, tokenizer) from a checkpoint file. Returns the raw checkpoint
# dict too, for any stage metadata stored alongside.
def load_checkpoint(path, map_location=None):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run the stage that produces it first "
            "(pretrain.py -> ckpt_base.pt, sft.py -> ckpt_sft.pt, "
            "lora.py -> ckpt_lora.pt, dpo.py -> ckpt_dpo.pt)"
        )
    loc = map_location or device
    ckpt = torch.load(path, map_location=loc)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(loc)
    model.load_state_dict(ckpt["model"])
    tok = WordTokenizer(ckpt["vocab"])
    return model, tok, ckpt


# Generate a continuation for a prompt and decode it (prompt sliced off).
def sample_story(model, tok, prompt="Once", n=80, temperature=0.8, top_k=50, eos_id=None):
    was_training = model.training
    model.eval()
    ids = tok.encode(prompt)
    dev = next(model.parameters()).device
    out = model.generate(torch.tensor([ids], device=dev), max_new_tokens=n,
                         temperature=temperature, top_k=top_k, eos_id=eos_id)
    model.train(was_training)
    return tok.decode(out[0].tolist()[len(ids):])  # continuation only
