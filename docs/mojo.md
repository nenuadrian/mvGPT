# Our GPT's kernels in Mojo

[Mojo](https://mojolang.org/) is Modular's language: it reads a lot like Python, it's compiled, with types, and it
writes GPU kernels in the language it uses for the CPU, compiling them for NVIDIA, AMD and, on a Mac, Metal. A Mojo
file can be imported from Python. `kernels.mojo` writes the path a token takes through our GPT as Mojo kernels, and
`mojo_model.py` calls them from Python, a token at a time, from one of our PyTorch checkpoints.

```bash
xcodebuild -downloadComponent MetalToolchain    # on a Mac: Mojo builds its kernels for Metal with it
uv pip install -r requirements.txt              # mojo and max, on macOS
python -m pytest tests/test_mojo.py             # skipped where Mojo isn't installed
python mojo_model.py checkpoints/sft/model.pt "Once upon a time"    # a story, and its tokens a second
python mojo_bench.py checkpoints/sft/model.pt   # writing, PyTorch against MLX against our kernels
mojo run launches.mojo                          # what launching a kernel and synchronizing cost
```

`mojo` is the compiler, with `mojo.importer`, the import hook that compiles a `.mojo` file the first time Python
imports it (into `__mojocache__/`); `max` is Modular's library, which holds Mojo's GPU modules (`max.gpu`).

## The kernels

`kernels.mojo` has episode one's blocks: learned positions, LayerNorm, GELU, biases, and a head tied to the token
embeddings. Its kernels are `embed`, `linear` (with `norm`, `gelu` and `add` as compile-time parameters, so each mix of
them compiles to its own kernel) and `attention` over the KV cache. `GPT` is a Mojo struct that Python makes and calls,
through `PythonModuleBuilder`: its init copies the weights to the GPU once, from one NumPy array of float32 in the
order `mojo_model.py`'s `flatten` lays them out, and its `step(token, pos, logits)` runs one token through the model
and writes the next token's logits into a NumPy array. `forward` queues the kernels on the GPU, then the copy of the
logits, and synchronizes once.

Mojo doesn't hand a plain `Int` to a GPU kernel (the kernels take `Int32`), refuses two writable pointers into one
buffer in a single call unless they have `MutAnyOrigin`, and asks a struct bound for Python for `write_to` and
`write_repr_to`.

`tests/test_mojo.py` checks the logits against PyTorch's, a token at a time (to 1e-4), and that greedy writing picks
PyTorch's tokens.

## Three versions

| Step | Kernels a token | How the work is shared |
|---|---|---|
| `ep035/step-2` | 57 | A thread for each number a kernel writes: a thread per row of a Linear layer, a thread per channel for LayerNorm, each repeating the mean and the variance |
| `ep035/step-4` | 57 | Warps of 32 threads: a warp per row, its lanes reading 4 neighbouring floats each, added up with `warp.sum`; a warp for LayerNorm; eight warps a head for attention |
| `ep035/step-6` | 32 | The LayerNorm inside the `linear` that reads it, each warp normalising its input on the way; q, k and v as one Linear writing a row of the cache |

`launches.mojo` measures the cost of starting kernels: on a Mac mini with an M4 Pro, handing a kernel to the GPU costs
the CPU about 7.6 µs, and 57 kernels that do next to nothing, then a synchronize, take about 710 µs (32 take about
410 µs). A token took about 1,130 µs after step 4, so about 63% of it was the cost of starting 57 kernels and waiting
for them.

## The race

`mojo_bench.py` writes 200 tokens greedily with PyTorch (MPS), MLX and our kernels, from one checkpoint, taking turns
with `mlx_bench.py`'s race, and checks that they write the same story. On episode three's fine-tuned model, in tokens a second, each version's
race recorded on its own:

| | PyTorch (MPS) | MLX | Our kernels |
|---|---|---|---|
| step 3: a thread per number | 496 | 1,227 | 287 |
| step 4: warps | 453 | 1,169 | 887 |
| step 6: fewer kernels | 493 | 1,231 | 1,103 |

All three wrote the same story each time. Warps did the most: a warp reads 128 neighbouring floats of a row at a time,
where a thread per row has the 32 threads of a warp reading floats a row apart. Fewer kernels took the last version to
about 2.2 times PyTorch's speed and nine tenths of MLX's. How much of what's left is MLX's kernels and how much its
launches would take Metal's profiler to tell.
