import math

import torch


def attention_by_hand(q, k, v):
    # Causal attention as the paper writes it, for q, k and v of (batch, heads, tokens, 64): every
    # query's score against every key, a table of tokens x tokens per head, masked so that each token
    # sees itself and the tokens before it, then the softmax of each row, and the values it weighs.
    T = q.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v
