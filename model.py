import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple silicon
        return "mps"
    return "cpu"


@dataclass
class GPTConfig:
    vocab_size: int = 10_000
    block_size: int = 256  # the context window, in tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384  # split between the heads: 6 x 64
    dropout: float = 0.1
    bias: bool = True
    rope: bool = False  # rotary position embeddings, instead of a learned embedding per position


class KVCache:
    def __init__(self, n_layer):
        # Per block: the keys and values of the tokens so far, each (B, heads, tokens, 64).
        self.keys = [None] * n_layer
        self.values = [None] * n_layer
        # The position of the next token, which, once the cache slides, is past the tokens it keeps.
        self.position = 0

    def __len__(self):
        return 0 if self.keys[0] is None else self.keys[0].size(2)

    def slide(self, keep):
        # Drop the oldest tokens' keys and values, keeping the last `keep`.
        self.keys = [k[:, :, -keep:] for k in self.keys]
        self.values = [v[:, :, -keep:] for v in self.values]

    def update(self, layer, k, v):
        if self.keys[layer] is not None:
            k = torch.cat([self.keys[layer], k], dim=2)
            v = torch.cat([self.values[layer], v], dim=2)
        self.keys[layer], self.values[layer] = k, v
        return k, v


def rotary(start, T, head_size, device):
    # The angles for positions start to start + T - 1, one per pair of channels, as cosines and sines:
    # they turn faster for the first pairs and slower for the last ones.
    speeds = 10_000 ** (-torch.arange(0, head_size, 2, device=device) / head_size)
    angles = torch.arange(start, start + T, device=device)[:, None] * speeds
    return angles.cos(), angles.sin()


def rotate(x, cos, sin):
    # Rotary position embedding: each pair of channels (i, i + 32) is turned by its position's angle.
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.n_head = config.n_head
        self.layer = layer
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        # Another function for causal attention over whole sequences, such as flash.py's kernel (GPT.attend_with).
        self.attend = None

    def forward(self, x, cache=None, angles=None):
        B, T, C = x.shape  # batch, tokens, channels
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if angles is not None:
            # Rotated by their positions, a query and a key score by how far apart they are, not where.
            q, k = rotate(q, *angles), rotate(k, *angles)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        S = k.size(2)  # the tokens attended to: the cached ones and these T
        # Query i is token S - T + i, which sees the keys up to its own. A single new token sees all S.
        mask = None if T in (1, S) else torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
        dropout_p = self.dropout_p if self.training else 0.0
        if self.attend is not None and S == T and dropout_p == 0.0:
            y = self.attend(q, k, v)
        else:
            # softmax(q k^T / sqrt(64)) v, where each token sees itself and the tokens before it.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=S == T)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))


def gelu(x):
    # x times the probability that a standard normal variable is at most x.
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(gelu(self.up_proj(x))))


class Block(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer)
        self.mlp_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cache=None, angles=None):
        x = x + self.attn(self.attn_norm(x), cache, angles)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        # With rotary embeddings, attention handles positions, and there's no table of them.
        self.embed_positions = None if config.rope else nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, layer) for layer in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the head scores tokens with the matrix that embeds them.
        self.embed_tokens.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                std = 0.02
                # These two add into the residual stream, twice per block, so they start smaller.
                if name.endswith(("o_proj", "down_proj")):
                    std /= math.sqrt(2 * self.config.n_layer)
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def attend_with(self, attend):
        # attend(q, k, v), causal, in place of PyTorch's attention where it can: whole sequences, no
        # dropout. None goes back to PyTorch's.
        for block in self.blocks:
            block.attn.attend = attend

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        # With a cache, these tokens come after the ones in it, so their positions do too.
        start = 0 if cache is None else cache.position
        x = self.embed_tokens(idx)
        angles = None
        if self.config.rope:
            # Computed once, for all the blocks.
            angles = rotary(start, T, self.config.n_embd // self.config.n_head, idx.device)
        else:
            x = x + self.embed_positions(torch.arange(start, start + T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, cache, angles)
        if cache is not None:
            cache.position += T
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            # A target of -1 marks a position with nothing to learn, such as padding.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None, use_cache=True):
        cache = None
        window = self.config.block_size
        for _ in range(max_new_tokens):
            # Past block_size tokens, the window slides, which moves each token to a new position and
            # changes its keys and values, so from there on we run the whole window again. With rotary
            # embeddings, the cache can slide instead, as a key's rotation stays right as the window moves.
            if not use_cache or (idx.size(1) > window and not self.config.rope):
                # The model has block_size positions, so it sees no more than the last 256 tokens.
                logits, _ = self(idx[:, -window:])
            elif cache is None:
                cache = KVCache(self.config.n_layer)
                logits, _ = self(idx[:, -window:], cache=cache)
            else:
                if len(cache) == window:
                    cache.slide(window - 1)
                # The newest token alone: the cache has the keys and values of the ones before it.
                logits, _ = self(idx[:, -1:], cache=cache)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits[logits < kth] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx
