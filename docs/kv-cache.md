# The KV cache

`generate` writes a token at a time. Without a cache, each step runs the whole story so far through
the model, all six blocks for each token, to read the logits at the last position. The tokens before
the new one come out as they did on the step before, so most of that work is done again for nothing:
writing 200 tokens after a 4-token prompt runs 4 + 5 + ... + 203 = 20,700 tokens through the model.
With a cache, it runs the prompt once and then each new token once, 203 tokens in all.

## What it keeps

Attention is causal: a token's channels, in each block, depend on that token and the ones before it,
and on none after it. So writing a new token leaves what was computed for the earlier tokens as it
was. For the newest token, $t$, attention needs its query, and the keys and values of the tokens up
to it:

$$
y_t = \text{softmax}\left(\frac{q_t\,[k_1, \dots, k_t]^\top}{\sqrt{d_k}}\right)[v_1, \dots, v_t]
$$

The keys and values of tokens $1$ to $t - 1$ are the ones the step before computed, so the cache
keeps them, a set per block, and each step computes the query, key and value of the new token alone
and adds its key and value. The earlier tokens' queries aren't kept, as a query is used to predict
the token after its own, and those have been predicted already: it's a KV cache, not a QKV cache.
The MLP, the norms and the head work on each token on its own, so they run on the new token alone
too.

## In the code

[`model.py`](https://github.com/nenuadrian/mvGPT/blob/main/model.py) has a `KVCache` with a slot per
block, for the keys and for the values, each `(batch, heads, tokens, 64)`. `len(cache)` is how many
tokens it holds, and `cache.update(layer, k, v)` joins the new tokens' keys and values after the
slot's, with `torch.cat` along the tokens, and returns all of them. Each `CausalSelfAttention` knows
its block's number, `layer`, to find its slot.

With a cache, attention's $T$ new queries attend over $S$ keys: the cached tokens' and their own.
The mask depends on the case:

| Case | Sizes | Mask |
| --- | --- | --- |
| No cache, as in training | $T = S$ | `is_causal=True`, as before |
| One new token | $T = 1$ | none: it sees all $S$ tokens |
| Several new tokens after cached ones | $1 < T < S$ | `torch.ones(T, S).tril(S - T)`: query $i$ is token $S - T + i$ |

The positions move too: the new tokens come after the cached ones, so `forward` starts their
positions at `len(cache)` instead of 0.

```python
logits, _ = model(prompt, cache=cache)       # the prompt fills the cache
logits, _ = model(next_id, cache=cache)      # then one token at a time
```

`generate` does this by default, so `sample.py`, `sft.py`'s samples and `evaluate.py` use the cache.
`use_cache=False` runs the whole context for each token, as before.

### The context window

The model has 256 learned positions. Once a story is longer than that, the window slides along it:
each token moves to a new position, which changes its channels from the first block on, so the
cached keys and values no longer hold. From there, `generate` runs the whole window for each token,
as it did without a cache.

## Checking and timing it

```bash
python bench.py checkpoints/sft/model.pt --device cpu
python bench.py checkpoints/sft/model.pt --device mps    # Apple's GPU; cuda on NVIDIA's
```

[`bench.py`](https://github.com/nenuadrian/mvGPT/blob/main/bench.py) first checks that the cache
changes nothing: the logits of a story run all at once and one token at a time through the cache
differ by about $2 \times 10^{-5}$, rounding from the arithmetic's different order, and a story
sampled from one seed comes out token for token as it did without the cache. Then it times each new
token as the story grows, and the tokens per second `generate` writes for one and eight stories.

On an Apple M4 Pro, with the fine-tuned model:

| | CPU, 1 story | CPU, 8 stories | GPU (`mps`), 1 story | GPU, 8 stories |
| --- | --- | --- | --- | --- |
| Without the cache | 169 tokens/s | 390 | 305 | 898 |
| With the cache | 704 tokens/s | 1,896 | 486 | 3,179 |
| Speed-up | 4.2x | 4.9x | 1.6x | 3.5x |

On the CPU, a token without the cache takes longer as the story grows, from about 2.8 ms to
8.7 ms by the end of the window, while with the cache it stays at about 1.5 ms. On the GPU,
both start at about 1.7 ms: for one story, a model this small gives the GPU too little work to
hide the cost of running its operations one after another, so a step on one token takes about as
long as a step on four. The cache saves the time above that floor, which is less than on the CPU.
With eight stories, each step has eight times the work for as many operations, and the GPU pulls
ahead.

## At scale

| | Per token | Full context |
| --- | --- | --- |
| mvGPT: 2 × 6 blocks × 384 × 4 bytes | 18,432 bytes | 4.7 MB (256 tokens) |
| Llama 3.1 70B: 2 × 80 blocks × 8 × 128 × 2 bytes | 327,680 bytes | 43 GB (131,072 tokens) |

Llama 3.1 70B's 64 query heads share 8 sets of keys and values (grouped-query attention), which
makes its cache 8 times smaller than with one set per head. Even so, at that scale the cache limits
how many conversations a GPU can serve at once, which is why serving systems such as
[vLLM](https://github.com/vllm-project/vllm) manage it in fixed-size pages, as an operating system
does with memory.
