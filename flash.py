import math

import torch
import triton
import triton.language as tl

# exp(x) = 2^(x log2 e), and a GPU raises 2 to a power faster than e: the kernels keep scores times log2 e.
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _forward(Q, K, V, O, LSE, scale, T, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
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
    # Each row's log-sum-exp, for the backward pass to make the probabilities again from the scores.
    tl.store(LSE + head * T + rows, m + tl.math.log2(l), mask=rows < T)
    tl.store(O + (head * T + rows[:, None]) * D + dims[None, :], (acc / l[:, None]).to(O.dtype.element_ty), mask=rows[:, None] < T)


@triton.jit
def _backward_kv(Q, K, V, DO, LSE, DELTA, DK, DV, scale, T, D: tl.constexpr, BLOCK: tl.constexpr):
    # A program for each block of keys and values, with the queries that see them going past in blocks.
    start = tl.program_id(0) * BLOCK
    head = tl.program_id(1).to(tl.int64)
    cols = start + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    k = tl.load(K + (head * T + cols[:, None]) * D + dims[None, :], mask=cols[:, None] < T, other=0.0)
    v = tl.load(V + (head * T + cols[:, None]) * D + dims[None, :], mask=cols[:, None] < T, other=0.0)
    dk = tl.zeros([BLOCK, D], tl.float32)
    dv = tl.zeros([BLOCK, D], tl.float32)
    # Causal: the queries from this block's first key on.
    for i in range(start, T, BLOCK):
        rows = i + tl.arange(0, BLOCK)
        q = tl.load(Q + (head * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
        do = tl.load(DO + (head * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
        lse = tl.load(LSE + head * T + rows, mask=rows < T, other=0.0)
        delta = tl.load(DELTA + head * T + rows, mask=rows < T, other=0.0)
        # The probabilities again, from the scores and the saved log-sum-exp: cheaper than storing them.
        p = tl.math.exp2(tl.dot(q, tl.trans(k)) * (scale * LOG2E) - lse[:, None])
        p = tl.where((rows[:, None] >= cols[None, :]) & (rows[:, None] < T), p, 0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # The softmax's backward: dS = P * (dP - the row's sum of dO * O).
        ds = p * (tl.dot(do, tl.trans(v)) - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    tl.store(DK + (head * T + cols[:, None]) * D + dims[None, :], (dk * scale).to(DK.dtype.element_ty), mask=cols[:, None] < T)
    tl.store(DV + (head * T + cols[:, None]) * D + dims[None, :], dv.to(DV.dtype.element_ty), mask=cols[:, None] < T)


@triton.jit
def _backward_q(Q, K, V, DO, LSE, DELTA, DQ, scale, T, D: tl.constexpr, BLOCK: tl.constexpr):
    # A program for each block of queries, with the keys they see going past in blocks.
    start = tl.program_id(0) * BLOCK
    head = tl.program_id(1).to(tl.int64)
    rows = start + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    q = tl.load(Q + (head * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
    do = tl.load(DO + (head * T + rows[:, None]) * D + dims[None, :], mask=rows[:, None] < T, other=0.0)
    lse = tl.load(LSE + head * T + rows, mask=rows < T, other=0.0)
    delta = tl.load(DELTA + head * T + rows, mask=rows < T, other=0.0)
    dq = tl.zeros([BLOCK, D], tl.float32)
    for j in range(0, tl.minimum(start + BLOCK, T), BLOCK):
        cols = j + tl.arange(0, BLOCK)
        k = tl.load(K + (head * T + cols[:, None]) * D + dims[None, :], mask=cols[:, None] < T, other=0.0)
        v = tl.load(V + (head * T + cols[:, None]) * D + dims[None, :], mask=cols[:, None] < T, other=0.0)
        p = tl.math.exp2(tl.dot(q, tl.trans(k)) * (scale * LOG2E) - lse[:, None])
        p = tl.where(rows[:, None] >= cols[None, :], p, 0.0)
        ds = p * (tl.dot(do, tl.trans(v)) - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + (head * T + rows[:, None]) * D + dims[None, :], (dq * scale).to(DQ.dtype.element_ty), mask=rows[:, None] < T)


class FlashAttention(torch.autograd.Function):
    # 128 queries against 64 keys at a time, forwards; 64 against 64 backwards, with more to hold.
    BLOCK_M, BLOCK_N, BLOCK = 128, 64, 64

    @staticmethod
    def forward(ctx, q, k, v):
        B, H, T, D = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o = torch.empty_like(q)
        lse = torch.empty(B * H, T, device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(T, FlashAttention.BLOCK_M), B * H)
        _forward[grid](q, k, v, o, lse, 1 / math.sqrt(D), T, D=D,
                       BLOCK_M=FlashAttention.BLOCK_M, BLOCK_N=FlashAttention.BLOCK_N, num_warps=4, num_stages=2)
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, T, D = q.shape
        do = do.contiguous()
        delta = (do.float() * o.float()).sum(-1).view(B * H, T)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        grid = (triton.cdiv(T, FlashAttention.BLOCK), B * H)
        args = dict(D=D, BLOCK=FlashAttention.BLOCK, num_warps=4, num_stages=3)
        _backward_kv[grid](q, k, v, do, lse, delta, dk, dv, 1 / math.sqrt(D), T, **args)
        _backward_q[grid](q, k, v, do, lse, delta, dq, 1 / math.sqrt(D), T, **args)
        return dq, dk, dv


def flash_attention(q, k, v):
    # Causal attention for q, k and v of (batch, heads, tokens, 64), with our kernels. Under autocast,
    # in its precision, as PyTorch's attention would be: rotary embeddings leave q and k in float32.
    if torch.is_autocast_enabled(q.device.type):
        dtype = torch.get_autocast_dtype(q.device.type)
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    return FlashAttention.apply(q, k, v)
