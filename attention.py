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


def flash_attention_pytorch(q, k, v, block=64):
    # The same attention without the table: FlashAttention's algorithm, a block of queries against a
    # block of keys at a time. The keys go past each block of queries in blocks, and a running maximum
    # and a running sum make the softmax come out as if all the scores had been there at once.
    T = q.size(-2)
    scale = 1 / math.sqrt(q.size(-1))
    out = torch.empty_like(q)
    for i in range(0, T, block):
        qi = q[:, :, i : i + block] * scale
        rows = torch.arange(i, i + qi.size(2), device=q.device)
        m = torch.full(qi.shape[:-1], float("-inf"), device=q.device)  # the largest score so far, per row
        l = torch.zeros(qi.shape[:-1], device=q.device)  # the sum of exp(score - m) so far
        acc = torch.zeros(qi.shape, device=q.device)  # the values, weighed by exp(score - m), so far
        # Causal: the queries in this block see no key past its last row.
        for j in range(0, i + qi.size(2), block):
            kj, vj = k[:, :, j : j + block], v[:, :, j : j + block]
            s = (qi @ kj.transpose(-2, -1)).float()
            cols = torch.arange(j, j + kj.size(2), device=q.device)
            s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            # A new maximum shrinks what was summed so far by exp(old - new): as if it had been there all along.
            correction = torch.exp(m - m_new)
            p = torch.exp(s - m_new[..., None])
            l = l * correction + p.sum(dim=-1)
            acc = acc * correction[..., None] + p @ vj.float()
            m = m_new
        out[:, :, i : i + block] = (acc / l[..., None]).to(q.dtype)
    return out
