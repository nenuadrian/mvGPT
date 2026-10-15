# Flash attention

Attention, as the paper writes it, scores every query against every key: for a story of $T$ tokens,
a table of $T \times T$ scores per head, masked, put through a softmax row by row, then multiplied
by the values. The table grows with the square of the length. At 256 tokens it's small; at 8,192
tokens it's 64 million scores per head, and in training, the table is kept for the backward pass, in
every block.

FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135), and
[FlashAttention-2](https://arxiv.org/abs/2307.08691)) computes the same attention without ever
storing the table. It goes through it a block at a time: a block of queries against a block of keys,
in a GPU's small, fast on-chip memory, and only the output goes back to the GPU's main memory. Most
of attention's time on a GPU goes to moving the table to and from that main memory, not to the
arithmetic, so skipping it is faster as well as smaller.

## The online softmax

The softmax of a row needs the row's largest score and the sum of all its exponentials, which seems
to need the whole row. The trick (from [Milakov and Gimelshein, 2018](https://arxiv.org/abs/1805.02867))
is to keep, for each row, a running maximum $m$, a running sum $\ell$ of $e^{s - m}$, and a running
sum of the values weighed by $e^{s - m}$. When a new block of scores $s$ comes in:

$$
m' = \max(m, \max_j s_j), \qquad
\ell' = \ell\, e^{m - m'} + \sum_j e^{s_j - m'}, \qquad
\mathrm{acc}' = \mathrm{acc}\, e^{m - m'} + \sum_j e^{s_j - m'} v_j
$$

A larger maximum shrinks what was summed so far by $e^{m - m'}$, as if it had been the maximum all
along. After the last block, the output is $\mathrm{acc} / \ell$: exactly the softmax's weighted sum
of the values, up to rounding. With causal attention, a block of queries stops at its last row: the
keys after it are masked anyway, which skips about half the blocks.

## In the code

- `attention.py`: `attention_by_hand`, attention with its table, and `flash_attention_pytorch`, the
  algorithm above in PyTorch, with Python loops over blocks of 64 queries and 64 keys. It runs
  anywhere, the Mac's GPU included, and shows the memory saved, but a loop in Python is slow.
- `flash.py`: the same loop as a Triton kernel. [Triton](https://triton-lang.org) is Python that
  compiles to an NVIDIA (or AMD) GPU; it runs on Linux. A program handles 128 queries of one head,
  going through the keys 64 at a time, and thousands of programs run at once. It works in powers of 2
  (`exp2`), which a GPU computes faster than powers of $e$, and saves each row's log-sum-exp,
  $m + \log \ell$.
- The backward pass makes the probabilities again from the scores and the log-sum-exp, a block at a
  time, rather than storing them: one kernel computes the gradients of a block of keys and values,
  with the queries that see them going past, and another those of a block of queries.
  `FlashAttention` is an autograd function with both.
- `GPT.attend_with(fn)` in `model.py` puts another causal attention in place of PyTorch's where it
  can, over whole sequences without dropout, and `pretrain_ddp.py --attention` picks PyTorch's
  (`pytorch`), ours by hand (`by-hand`), or ours in Triton (`triton`).

```bash
python -m pytest tests/test_attention.py                    # by hand and in PyTorch, against PyTorch's
python attention_bench.py --device cpu                      # time and peak memory, by length
python attention_bench.py --device mps                      # time on the Mac's GPU
python -m pytest tests/test_flash.py                        # the Triton kernel, on an NVIDIA GPU
python attention_bench.py --dtype bfloat16 --batch 4 --backward
torchrun --standalone --nproc_per_node 1 pretrain_ddp.py 5 --block 2048 --batch 12 --attention triton
```

`tests/test_flash.py` compares the kernel's output and gradients with PyTorch's attention in
float64: in bfloat16 and float16, ours may be off by rounding, but by no more than twice what
PyTorch's own attention is off by in the same precision, the test FlashAttention's authors use.
Without a GPU, `TRITON_INTERPRET=1` runs the kernels on the CPU, slowly, in Triton's interpreter.

## Which attention PyTorch runs

`F.scaled_dot_product_attention` picks an implementation for the device, and `attention_bench.py`
asks PyTorch's profiler which:

| Device | PyTorch runs |
| --- | --- |
| CPU | `_scaled_dot_product_flash_attention_for_cpu`: flash attention, for CPUs |
| Apple GPU (MPS) | `_scaled_dot_product_attention_math_for_mps`: attention as written, in one operation |
| NVIDIA GPU, bfloat16 | `_scaled_dot_product_flash_attention`, which launches `pytorch_flash::flash_fwd_kernel`: FlashAttention-2 |

So `pretrain_ddp.py`, in bfloat16 on NVIDIA GPUs, has trained with FlashAttention all along.

## On the Mac

An Apple M4 Pro, one sequence of 12 heads of 64 channels, in float32. The Mac's GPU doesn't say
how much memory it held at most, so memory is measured on the CPU, in a process of its own:

| Tokens | Most memory, by hand | PyTorch's (flash, CPU) | Flash, in PyTorch |
| --- | --- | --- | --- |
| 1,024 | 104 MB | 10 MB | 9 MB |
| 2,048 | 402 MB | 13 MB | 11 MB |
| 4,096 | 1,576 MB | 19 MB | 18 MB |
| 8,192 | 6,301 MB | 32 MB | 30 MB |

By hand, the memory goes up four times when the length doubles; without the table, twice. On the
Mac's GPU:

| Tokens | By hand | PyTorch's | Flash, in PyTorch |
| --- | --- | --- | --- |
| 1,024 | 6.8 ms | 1.2 ms | 19.5 ms |
| 2,048 | 10.1 ms | 1.9 ms | 59.5 ms |
| 4,096 | 37.5 ms | 6.8 ms | 236 ms |
| 8,192 | 171 ms | 27.1 ms | 1,143 ms |

The algorithm in Python launches a few small operations per pair of blocks, thousands of them, so
it's the slowest of the three: it shows the memory saved, and a kernel is what makes it fast.

## On an NVIDIA GPU

A rented RTX 4090 with 24 GB, four sequences of 12 heads of 64 channels, in bfloat16. Forwards:

| Tokens | By hand | PyTorch's (FlashAttention-2) | Ours, in Triton |
| --- | --- | --- | --- |
| 1,024 | 1.23 ms, 199 MB | 0.11 ms, 6 MB | 0.14 ms, 6 MB |
| 2,048 | 4.59 ms, 784 MB | 0.28 ms, 12 MB | 0.30 ms, 12 MB |
| 4,096 | 18.3 ms, 3,112 MB | 0.80 ms, 25 MB | 0.82 ms, 24 MB |
| 8,192 | 73.4 ms, 12,416 MB | 2.78 ms, 50 MB | 2.82 ms, 48 MB |
| 16,384 | out of memory | 10.5 ms, 99 MB | 10.6 ms, 96 MB |

At 16,384 tokens, ours does 156 trillion floating-point operations a second, PyTorch's 157, and by
hand, at 8,192 tokens, 5.6: most of its time goes to moving the table. Forwards and backwards, as in
training:

| Tokens | By hand | PyTorch's | Ours |
| --- | --- | --- | --- |
| 1,024 | 3.09 ms, 397 MB | 0.43 ms, 48 MB | 0.62 ms, 42 MB |
| 2,048 | 11.8 ms, 1,564 MB | 0.99 ms, 97 MB | 1.24 ms, 84 MB |
| 4,096 | 46.3 ms, 6,208 MB | 2.92 ms, 194 MB | 3.70 ms, 169 MB |
| 8,192 | out of memory | 10.3 ms, 387 MB | 13.2 ms, 338 MB |
| 16,384 | out of memory | 38.0 ms, 774 MB | 49.3 ms, 675 MB |

Backwards, ours is about three quarters as fast: each of its two kernels computes the scores again,
where FlashAttention-2 computes them once and adds up the queries' gradients across programs with
atomic additions.

Pretraining the 91-million-parameter model of the [several GPUs](distributed.md) page on the 4090,
with a context of 2,048 tokens, 12 sequences a step, five minutes each, from the same seed:

| | PyTorch's attention | Ours |
| --- | --- | --- |
| Tokens per second | 85,447 | 81,703 |
| Steps in five minutes | 827 | 788 |
| Held-out loss at steps 250, 500 and 750 | 2.537, 2.027, 1.837 | 2.540, 2.021, 1.844 |
| Peak memory | 16.2 GB | 16.2 GB |

The losses follow each other to the rounding of bfloat16, and attention is a small part of a step's
time, so the whole step is 4% slower. By hand, it runs out of memory before the first step, at the
first evaluation, which scores 48 sequences of 2,048 tokens at once: a table of 9 GiB, in float32.
