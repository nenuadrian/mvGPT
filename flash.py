import math

import torch
import triton
import triton.language as tl

# exp(x) = 2^(x log2 e), and a GPU raises 2 to a power faster than e: the kernels keep scores times log2 e.
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _forward(Q, K, V, O, scale, T, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # A program for each block of BLOCK_M queries of one head of one sequence; thousands run at once.
    # q, k, v and o are (sequences x heads, tokens, D), one after the other in memory.
    start = tl.program_id(0) * BLOCK_M
    head = tl.program_id(1).to(tl.int64)
    rows = start + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    # The block of queries, read once from the GPU's memory into the program's own, much faster one.
    q = tl.load(Q + (head * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
    m = tl.full([BLOCK_M], float("-inf"), tl.float32)  # the largest score so far, per row
    l = tl.zeros([BLOCK_M], tl.float32)  # the sum of 2^(score - m) so far
    acc = tl.zeros([BLOCK_M, D], tl.float32)  # the values, weighed by 2^(score - m), so far
    # Causal: the keys up to this block's last query, a block at a time.
    for j in range(0, tl.minimum(start + BLOCK_M, T), BLOCK_N):
        cols = j + tl.arange(0, BLOCK_N)
        k = tl.load(K + (head * T + cols[:, None]) * D + dims[None, :], mask=cols[:, None] < T, other=0.0)
        v = tl.load(V + (head * T + cols[:, None]) * D + dims[None, :], mask=cols[:, None] < T, other=0.0)
        s = tl.dot(q, tl.trans(k)) * (scale * LOG2E)
        s = tl.where(rows[:, None] >= cols[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        correction = tl.math.exp2(m - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l = l * correction + tl.sum(p, 1)
        acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    tl.store(O + (head * T + rows[:, None]) * D + dims[None, :], (acc / l[:, None]).to(O.dtype.element_ty), mask=rows[:, None] < T)


BLOCK_M, BLOCK_N = 128, 64


def flash_attention(q, k, v):
    # Causal attention for q, k and v of (batch, heads, tokens, 64), with our kernel: a program for each
    # block of 128 queries of each head of each sequence, going through the keys 64 at a time.
    B, H, T, D = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    grid = (triton.cdiv(T, BLOCK_M), B * H)
    _forward[grid](q, k, v, o, 1 / math.sqrt(D), T, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
    return o
