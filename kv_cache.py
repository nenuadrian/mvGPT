"""KV-cache inference: the same GPT as base_model, extended with a cache.

Why: base_model.GPT.generate() re-runs the ENTIRE context through all blocks
for every new token - to emit token T+1 it recomputes keys and values for
tokens 0..T that it already computed last step. That makes step t cost O(t),
and a whole generation O(T^2) forward work.

The fix: each attention layer returns the (k, v) it computed; generate()
keeps them and passes them back in. The next step feeds ONLY the new token -
the layer computes q,k,v for that one position, concatenates onto the cached
k,v, and attends over everything. Per-step cost drops from "whole prefix"
to "one token".

Two things must respect the cache:
  - position embeddings: the new token sits at absolute position past_len,
    not 0, so positions are offset by the cache length
  - causal mask: a single query at global position past_len may see all
    cached keys, so the mask row is sliced at the global row index

The parameter names/shapes are identical to base_model.GPT, so any
checkpoint from pretrain/sft/dpo loads here unchanged; only forward() differs.

Run:  python kv_cache.py      (loads ckpt_sft.pt, falls back to ckpt_base.pt)
"""

import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from base_model import (CKPT_BASE, CKPT_SFT, GPTConfig, MLP, WordTokenizer,
                        device)


# Same as base_model.CausalSelfAttention, plus: accepts the (k, v) computed on
# previous steps and returns the updated (k, v) for the caller to keep.
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # Lower-triangular boolean mask: True where attention is allowed.
        # Still needed here (unlike base_model): SDPA's is_causal assumes the
        # query at row i sits at global position i, which is wrong once a cache
        # offsets the queries, so we slice this mask and pass it explicitly.
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        )

    def forward(self, x, past_kv=None):           # x: (B, T, C); T=1 once the cache is warm
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv                       # (B, nh, past_len, hd)
            k = torch.cat([pk, k], dim=2)          # THE cache step: reuse, don't recompute
            v = torch.cat([pv, v], dim=2)

        S = k.size(2)                              # total keys = past_len + T
        # query i lives at global position S-T+i, so its mask row is S-T+i:
        # with a warm cache (T=1) that row lets it see every cached key.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.mask[S - T:S, :S],
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                          # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y)), (k, v)


# Same wiring as base_model.Block; just threads the cache through attention.
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None):
        y, kv = self.attn(self.attn_norm(x), past_kv)
        x = x + y
        x = x + self.mlp(self.mlp_norm(x))
        return x, kv


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.embed_positions = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.embed_tokens.weight = self.lm_head.weight

    # past_kvs: list of one (k, v) per block, or None to start fresh.
    # Returns the updated list so the caller can pass it back next step.
    def forward(self, idx, targets=None, past_kvs=None):
        B, T = idx.shape
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)
        # the new tokens continue the sequence, so positions start at past_len
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.embed_tokens(idx) + self.embed_positions(pos))

        new_kvs = []
        for block, past in zip(self.blocks, past_kvs or [None] * len(self.blocks)):
            x, kv = block(x, past)
            new_kvs.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None,
                 use_cache=True):
        past = None
        x = idx  # first step: the whole prompt fills the cache in one forward
        for _ in range(max_new_tokens):
            if idx.size(1) >= self.cfg.block_size:
                break  # cached keys hold absolute positions; stop at the context limit
            if use_cache:
                logits, _, past = self(x, past_kvs=past)
            else:
                logits, _, _ = self(idx)   # the O(T^2) baseline: re-run everything
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_id], dim=1)
            x = next_id  # warm cache: ONLY the new token goes through the model
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx


# Any pipeline checkpoint loads into this GPT - the state dicts match.
def load_cached_model(path):
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    tok = WordTokenizer(ckpt["vocab"])
    return model.eval(), tok  # eval: dropout off, outputs deterministic given seeds


def _sync():
    # wall-clock timing on an async backend needs a sync point
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    path = CKPT_SFT if os.path.exists(CKPT_SFT) else CKPT_BASE
    if not os.path.exists(path):
        raise FileNotFoundError("no checkpoint found - run pretrain.py (and ideally sft.py) first")
    print(f"loading {path}")
    model, tok = load_cached_model(path)

    # --- correctness: incremental cached forward == one full forward ---
    # Feed a prompt token by token through the cache and compare the logits
    # against processing it in a single batch. Same math, different batching,
    # so the difference is float noise (~1e-5), not exact zero.
    prompt = "Once upon a time there was a little girl who loved to play in the park"
    ids = torch.tensor([tok.encode(prompt)], device=device)

    with torch.no_grad():
        full_logits, _, _ = model(ids)
        past, steps = None, []
        for t in range(ids.size(1)):
            lg, _, past = model(ids[:, t:t + 1], past_kvs=past)
            steps.append(lg)
        inc_logits = torch.cat(steps, dim=1)

    diff = (full_logits - inc_logits).abs().max().item()
    print(f"max |full - incremental| logit diff over {ids.size(1)} positions: {diff:.2e}")

    # --- speed: same generation with and without the cache ---
    gen_prompt = torch.tensor([tok.encode("Once upon a time")], device=device)
    n = 200
    print(f"\ngenerating {n} tokens (batch 1):")
    for use_cache in (False, True):
        torch.manual_seed(0)
        _sync()
        t0 = time.time()
        out = model.generate(gen_prompt, max_new_tokens=n, temperature=0.8, top_k=50,
                             use_cache=use_cache)
        _sync()
        dt = time.time() - t0
        n_new = out.size(1) - gen_prompt.size(1)
        label = "with cache   " if use_cache else "without cache"
        print(f"  {label}: {n_new} tokens in {dt:5.2f}s  ({n_new / dt:6.1f} tok/s)")

    print("\nsample (cached):")
    print(tok.decode(out[0].tolist()[gen_prompt.size(1):]))


if __name__ == "__main__":
    main()
