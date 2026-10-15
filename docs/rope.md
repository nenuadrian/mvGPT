# Rotary position embeddings

mvGPT learned an embedding per position: a table of 256 vectors, added to the tokens' embeddings.
That has two limits. The model can't see past position 255, and, as the [KV cache](kv-cache.md)
page shows, once a story is longer than the window, each token moves to a new position, which
changes its keys and values, so the cache can't slide along with the window.

Rotary position embeddings (RoPE), from the RoFormer paper and used by Llama and most models since,
put positions into attention instead. Each head's query and key are split into pairs of channels,
and each pair is rotated by an angle proportional to the token's position:

$$
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix},
\qquad \theta_i = 10000^{-2i/d}
$$

for a token at position $m$. The dot product of a query at position $m$ and a key at position $n$
then depends on $m - n$ alone: rotating both by the same extra angle leaves their dot product as it
was. So attention sees how far apart two tokens are, and not where they are, and a key cached at one
position stays right when the window moves on.

## In the code

`GPTConfig(rope=True)` builds a model with no table of positions, 98,304 fewer parameters, and
`rotate` in `model.py` turns the queries and keys in each block, before they go into the cache. (It
pairs channel $i$ with channel $i + 32$, the "rotate half" form Llama's code uses, which is the
formula above with the channels in another order.) Checkpoints without `rope` in their config load
as before, with learned positions.

`KVCache` keeps the position of the next token apart from how many tokens it holds, and
`cache.slide(keep)` drops the oldest tokens' keys and values. With rotary embeddings, `generate`
slides the cache past the window, a token at a time, where a model with learned positions runs the
whole window again for each token.

```bash
python pretrain.py 10 --rope    # to checkpoints/pretrain-rope
python evaluate.py checkpoints/pretrain/model.pt checkpoints/pretrain-rope/model.pt
python context.py checkpoints/pretrain/model.pt checkpoints/pretrain-rope/model.pt
```

## Past the window

Ten minutes of pretraining each, on an Apple M4 Pro, from the same seed and data:

| | Learned positions | RoPE |
| --- | --- | --- |
| Parameters | 14,585,856 | 14,487,552 |
| Steps in 10 minutes | 1,250 | 1,178 |
| Held-out loss at step 1,000 | 2.31 | 2.18 |
| Held-out loss at the end | 2.25 | 2.18 |
| Perplexity on held-out stories (`evaluate.py`) | 9.5 | 8.5 |

RoPE learns faster per step, and ends lower despite fewer steps, as rotating the queries and keys
adds small operations to each block.

`context.py` measures tokens 256 to 767 of eight held-out streams of 768 tokens:

| Model, method | Loss | ms per token, CPU | ms per token, GPU |
| --- | --- | --- | --- |
| Learned positions, sliding window: the last 256 tokens, run in full for each token | 2.32 | 8.6 | 3.5 |
| RoPE, sliding window | 2.18 | 11.6 | 4.0 |
| RoPE, sliding cache: a token at a time, dropping the oldest | 2.18 | 2.2 | 3.0 |
| RoPE, all 768 tokens at once | 2.34 | | |

The sliding cache matches the sliding window's loss, and on the CPU, it's about five times faster
per token. On the GPU, with one sequence, a step is mostly overhead either way, as the
[profiling](profiling.md) page shows. Attending to all 768 tokens at once is worse: rotary
embeddings make attention depend on distances, but the model only ever saw distances up to 255, so
RoPE doesn't stretch the context the model was trained on. Models that do, such as Llama's longer
versions, train on longer sequences, or scale the rotations.
