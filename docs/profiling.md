# Profiling

Before deciding how to make the model faster, it's worth measuring where its time goes, as the
answer depends on the device. Two tools: Google's Model Explorer shows the model as a graph of the
operations PyTorch runs, and PyTorch's profiler times them.

## The model as a graph

[`explore.py`](https://github.com/nenuadrian/mvGPT/blob/main/explore.py) exports the model with
`torch.export`, which runs it on a full context window and records each operation, with its shapes,
as a graph: 163 operations for 256 tokens. It opens the graph in
[Model Explorer](https://github.com/google-ai-edge/model-explorer), in the browser.

```bash
python explore.py checkpoints/sft/model.pt            # http://localhost:8080
python explore.py checkpoints/sft/model.pt --times    # each operation coloured by its time on the CPU
```

With `--times`, it runs the graph a node at a time with `torch.fx.Interpreter`, each node labelled
by its name for the profiler, and gives the times to Model Explorer as node data. The head, which
scores the 10,000 tokens of the vocabulary at each position, is the slowest node.

## The profiler

[`profiler.py`](https://github.com/nenuadrian/mvGPT/blob/main/profiler.py) profiles a pass over a
full context window, where each operation has enough work for its time to be its own, and writing
100 tokens with the KV cache, where the operations are small and many:

```bash
python profiler.py checkpoints/sft/model.pt --device cpu --out results
python profiler.py checkpoints/sft/model.pt --device mps --out results
```

It prints each operation the run called, how often, and its share of the time, and writes the
pass's trace, which [Perfetto](https://ui.perfetto.dev) opens as a timeline. On an Apple M4 Pro's
CPU, the linear layers take about 60% of a pass, attention about 9%, and the GELU we write out, its
divisions, error functions and multiplications, about 11%. Writing, the model runs about 189
operations per token.

On a Mac's GPU, the profiler sees the CPU's side: the time to hand each operation over, as the GPU
runs it afterwards. A profiled pass takes about 1.6 ms to hand over, while it takes about 4.6 ms to
run. While writing, about 40% of the time shows up under `aten::to`, a copy of a single number,
which is where the CPU first waits for the GPU in each step, in top k's `values[:, [-1]]`. Remove
it, and the wait moves to the next operation that needs the GPU. On NVIDIA GPUs, the profiler also
records the GPU's kernels.

## Decisions, measured

[`decisions.py`](https://github.com/nenuadrian/mvGPT/blob/main/decisions.py) runs `model.py` with
a function swapped for another, through `unittest.mock`, and times a pass and a token:

| Decision | Pass, CPU | Pass, GPU | Token, CPU | Token, GPU |
| --- | --- | --- | --- | --- |
| As written | 8.8 ms | 4.6 ms | 1.30 ms | 2.06 ms |
| GELU as one operation, `F.gelu` | 7.8 ms | 3.6 ms | 1.26 ms | 1.97 ms |
| Attention written by hand | 10.6 ms | 4.7 ms | 1.40 ms | 5.09 ms |
| `torch.compile` | 8.9 ms | 3.8 ms | | |

| Vocabulary | Head | Pass, CPU | Pass, GPU |
| --- | --- | --- | --- |
| 5,000 | 1.9M | 8.4 ms | 3.5 ms |
| 10,000 | 3.8M | 8.7 ms | 3.7 ms |
| 20,000 | 7.7M | 9.2 ms | 4.4 ms |

GELU as one operation saves about a tenth of a pass on the CPU and a fifth on the GPU, as it's 24
fewer operations per token, each reading and writing the MLP's channels. `torch.compile` fuses
operations and helps on the GPU, where each operation has a cost of its own, and not on the CPU.
Attention written out is slower on both, and more than twice as slow for a token on the GPU.

`model.py` keeps its GELU written out, for reading, and `torch.compile` can fuse it where speed
matters.
