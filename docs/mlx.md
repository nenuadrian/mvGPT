# Our GPT in MLX

[MLX](https://ml-explore.github.io/mlx/) is Apple's array library for its own chips, from Apple's machine learning
research team (December 2023): NumPy-like arrays that live in memory the CPU and the GPU share, computed lazily, with
`mlx.nn` for layers, `mlx.optimizers`, function transformations for gradients and a compiler, `mx.compile`.
`mlx_model.py` writes our GPT in it, and `mlx_bench.py` races it against PyTorch on the same Mac.

```bash
uv pip install -r requirements.txt              # mlx and mlx-metal, on macOS
python -m pytest tests/test_mlx.py              # skipped where MLX isn't installed
python mlx_bench.py checkpoints/sft/model.pt    # writing and training, MLX against PyTorch
python mlx_pretrain.py 1250                     # pretraining in MLX, saved as checkpoints/pretrain-mlx/model.pt
python sample.py checkpoints/pretrain-mlx/model.pt "Once upon a time"
```

## The model

`mlx_model.py` has episode one's blocks (learned positions, LayerNorm, GELU, biases, a head tied to the token
embeddings) under the names `model.py` gives them, so `from_torch` loads one of our PyTorch checkpoints by name: each
tensor goes through NumPy into an MLX array, and `load_weights` raises an error for a name that doesn't match. MLX's
`Linear` stores its weight outputs by inputs, as PyTorch does, so nothing is transposed. Modules have `__call__` where
PyTorch has `forward`, attention uses `mx.fast.scaled_dot_product_attention` with `mask="causal"`, and the head is
`embed_tokens.as_linear`. A config with RoPE, RMSNorm, SwiGLU or grouped-query attention is refused.

`generate` keeps a list per layer as the cache and writes a token at a time; `item()` is where the GPU does the work,
as MLX builds the work up lazily until a value is needed. `to_torch` is the way back, for a model trained in MLX.

`tests/test_mlx.py` checks the logits against PyTorch's (to 1e-4), the greedy tokens with the cache, and that a model
goes to MLX and back with each weight unchanged.

## The race

`mlx_bench.py` writes 200 tokens with each, greedily, from one checkpoint, and checks they write the same story; then
it times training steps of episode two's model on `pretrain.py`'s batch with PyTorch, MLX, and MLX's compiled step, all
without dropout, as MLX's fused attention has none. The contenders take turns, a warm-up run each and then rounds, and
the fastest time of each is kept, so whatever else the machine runs slows them alike.

On a Mac mini with an M4 Pro, with other programs using the machine at the time (which is why they take turns), on
episode three's fine-tuned model:

| | PyTorch (MPS) | MLX | MLX, compiled |
|---|---|---|---|
| Writing 200 tokens, one at a time | 0.41 s, 486 tokens/s | 0.16 s, 1,276 tokens/s | |
| A training step, 48 x 256 tokens | 433 ms | 355 ms | 307 ms |

Both wrote the same story. Writing a token at a time is mostly the overhead of handing work to the GPU, where MLX's is
lower; a training step is mostly large matrix multiplies, and compiling the step makes about two fifths of MLX's lead.

## Training in MLX

`mlx_pretrain.py` is `pretrain.py`'s run for a number of steps: its batch, learning rates and warmup, the cosine
written with MLX's operations, AdamW with weight decay on the matrices only (`optim.MultiOptimizer`), gradients clipped
to 1, and the whole step compiled. Two things differ from `pretrain.py`: MLX's fused attention has no dropout (the rest
of the model has it), and MLX's AdamW leaves out Adam's bias correction by default. There's no `backward()`:
`nn.value_and_grad` gives a function that returns the loss and the gradients, and `optimizer.update` applies them. The
compiled step names the model's, the optimiser's and the random state as its inputs and outputs, and each step ends
with `mx.eval` on them.

1,250 steps (15.4M tokens) in 10.6 minutes, about 36,700 tokens a second after a slow start, on the shared GPU. The
held-out loss ended at 2.295, against 2.252 for episode two's PyTorch run of as many steps, a little higher at each
evaluation. MLX's layers and embedding start from their own initialisation, wider than `model.py`'s (0.02, smaller for
the projections into the residual stream), which isn't ported here; that's the likeliest reason, as less dropout should
help a run this short, and the bias correction matters in the first few dozen steps, inside the warmup.
