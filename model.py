"""mvGPT: a minimum viable GPT, a decoder-only transformer you can read end to end."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 10_000
    block_size: int = 256  # the context window, in tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384  # split between the heads: 6 x 64
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, tokens, channels
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # softmax(q k^T / sqrt(64)) v, where each token sees itself and the tokens before it.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True
        )
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
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.embed_positions = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.embed_tokens(idx) + self.embed_positions(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        for _ in range(max_new_tokens):
            # The model has block_size positions, so it sees no more than the last 256 tokens.
            logits, _ = self(idx[:, -self.config.block_size :])
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
