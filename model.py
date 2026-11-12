import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.nn import functional as F

from hyena import Hyena


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
    rope_scaling: str | None = None  # rotations stretched past the window trained on: "pi", "ntk", "parts" or "yarn"
    rope_factor: float = 1.0  # by how much: the window trained on is block_size / rope_factor
    causal: bool = True  # each token sees the ones before it; for diffusion (diffusion.py), all of them
    rmsnorm: bool = False  # RMSNorm in place of LayerNorm: each token's channels rescaled, not centred
    swiglu: bool = False  # a gated MLP, SwiGLU, in place of GELU's
    qk_norm: bool = False  # RMSNorm on each head's queries and keys, before attention compares them
    n_kv_head: int | None = None  # heads of keys and values, each shared by a group of query heads; None is n_head
    mixer: str = "attention"  # what mixes the tokens in each block: "attention", or "hyena" (hyena.py)
    attention_layers: tuple = ()  # with mixer "hyena", the blocks that keep attention, from 0: a hybrid
    n_expert: int = 0  # a mixture of experts in place of each block's MLP, with this many; 0 is one MLP
    expert_top_k: int = 2  # the experts each token goes to
    expert_hidden: int | None = None  # an expert's width; None is the MLP's divided by top k, for as much compute


# A modern block: RMSNorm, SwiGLU and no biases, as in LLaMA, and QK-norm, as in OLMo 2 and Qwen3. RoPE,
# which they use too, has a flag of its own.
MODERN = dict(rmsnorm=True, swiglu=True, qk_norm=True, bias=False)


class KVCache:
    def __init__(self, n_layer):
        # Per block: the keys and values of the tokens so far, each (B, key and value heads, tokens, 64).
        self.keys = [None] * n_layer
        self.values = [None] * n_layer
        # Hyena's blocks (hyena.py) keep theirs here instead: each token's gated values, and the last two projections.
        self.hyena = [None] * n_layer
        # The position of the next token, which, once the cache slides, is past the tokens it keeps.
        self.position = 0

    def __len__(self):
        for k in self.keys:
            if k is not None:
                return k.size(2)
        for state in self.hyena:
            if state is not None:
                return state["z"].size(-1)
        return 0

    def slide(self, keep):
        # Drop the oldest tokens' keys and values, keeping the last `keep`.
        self.keys = [k if k is None else k[:, :, -keep:] for k in self.keys]
        self.values = [v if v is None else v[:, :, -keep:] for v in self.values]
        self.hyena = [s if s is None else {**s, "z": s["z"][..., -keep:]} for s in self.hyena]

    def trim(self, n):
        # Drop the newest n tokens' keys and values: the guesses speculative decoding didn't keep.
        if n:
            self.keys = [k[:, :, :-n] for k in self.keys]
            self.values = [v[:, :, :-n] for v in self.values]
            self.position -= n

    def update(self, layer, k, v):
        if self.keys[layer] is not None:
            k = torch.cat([self.keys[layer], k], dim=2)
            v = torch.cat([self.values[layer], v], dim=2)
        self.keys[layer], self.values[layer] = k, v
        return k, v


def kv_cache_bytes(config, tokens, bytes_per_number=4):
    # Per token, a key and a value for each layer, key and value head and channel; float32 is 4 bytes a number.
    # A Hyena block keeps one number a channel instead, and its last two tokens' projections, which don't grow.
    hyena = config.n_layer - len(config.attention_layers) if config.mixer == "hyena" else 0
    per_token = 2 * (config.n_layer - hyena) * (config.n_kv_head or config.n_head) * (config.n_embd // config.n_head)
    return (per_token + hyena * config.n_embd) * tokens * bytes_per_number


SCALINGS = ("pi", "ntk", "parts", "yarn")
# NTK-by-parts and YaRN: a pair of channels that turns more than 32 times in the window trained on keeps its
# speed, one that turns less than once is slowed by the factor, and those in between are slowed by some.
FAST_TURNS, SLOW_TURNS = 32, 1


def rope_speeds(head_size, device, scaling=None, factor=1.0, trained=None):
    # A speed for each pair of channels, in radians per position: faster for the first pairs, slower for the last.
    speeds = 10_000 ** (-torch.arange(0, head_size, 2, device=device) / head_size)
    if scaling == "pi":
        # Position interpolation: every pair slowed by the factor, so position 4,096 turns as far as 1,024 did.
        return speeds / factor
    if scaling == "ntk":
        # NTK-aware: a bigger base, which leaves the fastest pair as it is and slows the slowest by the factor.
        base = 10_000 * factor ** (head_size / (head_size - 2))
        return base ** (-torch.arange(0, head_size, 2, device=device) / head_size)
    if scaling in ("parts", "yarn"):
        turns = trained * speeds / (2 * math.pi)  # in the window trained on
        keep = ((turns - SLOW_TURNS) / (FAST_TURNS - SLOW_TURNS)).clamp(0, 1)
        return keep * speeds + (1 - keep) * speeds / factor
    return speeds


def rotary(start, T, head_size, device, scaling=None, factor=1.0, trained=None):
    # The angles for positions start to start + T - 1, one per pair of channels, as cosines and sines.
    speeds = rope_speeds(head_size, device, scaling, factor, trained)
    angles = torch.arange(start, start + T, device=device)[:, None] * speeds
    cos, sin = angles.cos(), angles.sin()
    if scaling == "yarn":
        # YaRN's temperature: queries and keys both lengthened by m, so attention's scores grow by m squared,
        # a sharper softmax over the longer context.
        m = 0.1 * math.log(factor) + 1
        cos, sin = cos * m, sin * m
    return cos, sin


def stretch(config, scaling, factor):
    # The same weights, with a window factor times as long, and the rotations stretched to fit it.
    return replace(config, block_size=int(config.block_size * factor), rope_scaling=scaling, rope_factor=factor)


def rotate(x, cos, sin):
    # Rotary position embedding: each pair of channels (i, i + 32) is turned by its position's angle.
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.eps = eps
        # A gain per channel, starting at 1, and no bias.
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x):
        # In float32, as squaring in bfloat16 loses precision, and then back to x's type.
        x32 = x.float()
        # Each token's channels divided by their root mean square, which brings it to 1, with eps against dividing by 0.
        y = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight

    def extra_repr(self):
        # What printing the model shows in the brackets, as LayerNorm shows its size.
        return f"{self.weight.numel()}, eps={self.eps}"


def norm_layer(config, size):
    # LayerNorm subtracts the mean of a token's channels and divides by their spread; RMSNorm skips the mean.
    return RMSNorm(size) if config.rmsnorm else nn.LayerNorm(size, bias=config.bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.n_head = config.n_head
        # Grouped-query attention: n_kv_head heads of keys and values, each shared by a group of query heads.
        self.n_kv_head = config.n_kv_head or config.n_head
        if self.n_head % self.n_kv_head:
            raise ValueError(f"n_head ({self.n_head}) must be a multiple of n_kv_head ({self.n_kv_head})")
        self.layer = layer
        self.causal = config.causal
        head_size = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * head_size, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * head_size, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Queries and keys rescaled to a root mean square of 1, so the scores don't grow with their projections' weights.
        self.q_norm = RMSNorm(head_size) if config.qk_norm else None
        self.k_norm = RMSNorm(head_size) if config.qk_norm else None
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        # Another function for causal attention over whole sequences, such as flash.py's kernel (GPT.attend_with).
        self.attend = None
        # For looking inside the model (interpret.py): keep the attention weights, and turn heads off.
        self.keep, self.weights, self.off = False, None, []

    def forward(self, x, cache=None, angles=None):
        B, T, C = x.shape  # batch, tokens, channels
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, C // self.n_head).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if angles is not None:
            # Rotated by their positions, a query and a key score by how far apart they are, not where.
            q, k = rotate(q, *angles), rotate(k, *angles)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        if self.n_kv_head != self.n_head:
            # The cache keeps the fewer heads; attention needs one per query head, so each is repeated for its group.
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        S = k.size(2)  # the tokens attended to: the cached ones and these T
        # Query i is token S - T + i, which sees the keys up to its own. A single new token sees all S.
        mask = None if T in (1, S) else torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
        dropout_p = self.dropout_p if self.training else 0.0
        if self.keep:
            # Written out, as episode 13's attention_by_hand is, to keep the weights: for each token, how much it takes
            # from each token before it.
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
            seen = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T) if self.causal else None
            if seen is not None:
                scores = scores.masked_fill(~seen, float("-inf"))
            self.weights = scores.softmax(dim=-1)
            y = self.weights @ v
        elif self.attend is not None and self.causal and S == T and dropout_p == 0.0:
            y = self.attend(q, k, v)
        else:
            # softmax(q k^T / sqrt(64)) v, where each token sees itself and the tokens before it, or, when
            # the model isn't causal, every token in the sequence, before it and after.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p,
                                               is_causal=self.causal and S == T)
        if self.off:
            # A head turned off adds nothing to the residual stream.
            y = y.clone()
            y[:, self.off] = 0
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))


def gelu(x):
    # x times the probability that a standard normal variable is at most x.
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x):
    # x times its sigmoid: close to x when x is large, close to 0 when it's very negative, and smooth between.
    return x * torch.sigmoid(x)


def mlp_hidden(config):
    # SwiGLU has three matrices to GELU's two, so at 8/3 of the embedding size wide, it has as many weights,
    # rounded up to a multiple of 64, a size GPUs handle well: 1,024 for our 384.
    return 64 * math.ceil(8 * config.n_embd / 3 / 64) if config.swiglu else 4 * config.n_embd


class MLP(nn.Module):
    def __init__(self, config, hidden=None):
        super().__init__()
        hidden = hidden or mlp_hidden(config)
        # The gate: a second projection up, which scales each of up_proj's channels on the way through.
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias) if config.swiglu else None
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.gate_proj is None:
            return self.dropout(self.down_proj(gelu(self.up_proj(x))))
        return self.dropout(self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x)))


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.swiglu:
            raise ValueError("the experts are SwiGLU MLPs, as in the modern block: set swiglu")
        self.n_expert = config.n_expert
        self.top_k = config.expert_top_k
        # Each expert is an MLP of its own, and a token goes through top_k of them. At the MLP's width divided by
        # top_k, a token costs what it cost with one MLP, while the block holds n_expert / top_k times the weights.
        hidden = config.expert_hidden or mlp_hidden(config) // config.expert_top_k
        # The experts' weights, stacked: expert e's three projections are gate[e], up[e] and down[e], so that one
        # batched matrix multiply runs all the experts at once, instead of one small multiply per expert.
        E, C = config.n_expert, config.n_embd
        self.gate = nn.Parameter(torch.randn(E, C, hidden) * 0.02)
        self.up = nn.Parameter(torch.randn(E, C, hidden) * 0.02)
        # Started smaller, like the MLP's down_proj, as it adds into the residual stream.
        self.down = nn.Parameter(torch.randn(E, hidden, C) * 0.02 / math.sqrt(2 * config.n_layer))
        self.dropout = nn.Dropout(config.dropout)
        # The router scores each expert for each token, from the token's channels.
        self.router = nn.Linear(config.n_embd, config.n_expert, bias=False)
        # Added to the scores to pick the experts, and nowhere else: DeepSeek-V3's balancing without a loss, which
        # update_bias nudges after each optimiser step. It stays at zeros with the balancing loss.
        self.register_buffer("bias", torch.zeros(config.n_expert))
        self.aux_loss = None  # the last batch's balancing loss
        self.load = None  # and how many of its tokens went to each expert
        self.chosen = None  # and each token's experts, the first choice first

    def expert(self, e, x):
        # Expert e on its own, as an MLP runs: for the tests, and to read.
        return (silu(x @ self.gate[e]) * (x @ self.up[e])) @ self.down[e]

    def forward(self, x):
        B, T, C = x.shape
        x = x.reshape(B * T, C)
        probs = F.softmax(self.router(x), dim=-1)
        # Each token's top_k experts, by score with the bias, and their weights from the scores without it, scaled
        # to add up to 1.
        chosen = torch.topk(probs + self.bias, self.top_k, dim=-1).indices
        weights = probs.gather(1, chosen)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Sort the (token, expert) pairs by expert, so each expert gets its tokens as one block of rows.
        flat = chosen.reshape(-1)
        order = flat.argsort()
        expert = flat[order]
        token = order // self.top_k
        weight = weights.reshape(-1)[order]
        counts = torch.bincount(flat, minlength=self.n_expert)
        # Each pair's row within its expert's block: its place in the sorted list, minus where its expert starts.
        slot = torch.arange(flat.numel(), device=x.device) - (counts.cumsum(0) - counts)[expert]
        # One block per expert, as many rows as the busiest expert has tokens, the rest left as zeros.
        rows = x.new_zeros(self.n_expert, int(counts.max()), C)
        rows[expert, slot] = x[token]
        y = torch.bmm(silu(torch.bmm(rows, self.gate)) * torch.bmm(rows, self.up), self.down)
        # Each output scaled by its weight, and added into the row of the token it came from.
        out = torch.zeros_like(x).index_add_(0, token, y[expert, slot] * weight[:, None])

        # Switch Transformer's balancing loss: the share of the tokens each expert got, times its average score,
        # summed and times n_expert. Spread evenly, it's 1, and it grows as the router favours a few experts.
        share = counts.float() / flat.numel()
        self.aux_loss = self.n_expert * (share * probs.mean(dim=0)).sum()
        self.load = counts.detach()
        self.chosen = chosen.detach()
        return self.dropout(out).view(B, T, C)

    @torch.no_grad()
    def update_bias(self, rate=1e-3):
        # An expert that got more than its share of the last batch is picked a little less often from now on, and
        # one that got less, a little more: a step of rate either way, as in the DeepSeek-V3 paper.
        load = self.load.float()
        self.bias += rate * torch.sign(load.mean() - load)


class Block(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.attn_norm = norm_layer(config, config.n_embd)  # before attention, or before Hyena in its place
        if config.mixer == "hyena" and layer not in config.attention_layers:
            self.hyena = Hyena(config, layer)
        else:
            self.attn = CausalSelfAttention(config, layer)
        self.mlp_norm = norm_layer(config, config.n_embd)
        self.mlp = MoE(config) if config.n_expert else MLP(config)

    def forward(self, x, cache=None, angles=None):
        if hasattr(self, "hyena"):
            # Hyena's convolutions know how far back each token is, so the rotary angles aren't for them.
            x = x + self.hyena(self.attn_norm(x), cache)
        else:
            x = x + self.attn(self.attn_norm(x), cache, angles)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.rope_scaling is not None and (not config.rope or config.rope_scaling not in SCALINGS):
            raise ValueError(f"rope_scaling stretches rotary embeddings: one of {SCALINGS}, with rope=True")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        # With rotary embeddings, attention handles positions, and there's no table of them.
        self.embed_positions = None if config.rope else nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, layer) for layer in range(config.n_layer)])
        self.norm = norm_layer(config, config.n_embd)
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
            if hasattr(block, "attn"):
                block.attn.attend = attend

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        # With a cache, these tokens come after the ones in it, so their positions do too.
        start = 0 if cache is None else cache.position
        x = self.embed_tokens(idx)
        angles = None
        if self.config.rope:
            # Computed once, for all the blocks. A stretched model was trained on a window rope_factor times shorter.
            c = self.config
            angles = rotary(start, T, c.n_embd // c.n_head, idx.device, c.rope_scaling, c.rope_factor,
                            c.block_size / c.rope_factor)
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
