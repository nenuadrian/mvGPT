# mvGPT

**Minimum Viable GPT** is a small decoder-only transformer, written from scratch in PyTorch, that you
can read from one end to the other. It has about 14.6 million parameters, and it's trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a dataset of short children's
stories, so it learns to write believable English in minutes on a laptop.

It's the code of the YouTube series [Minimum Viable GPT](https://mvgpt.nenuadrian.com), which builds
it one tagged step at a time and then takes it through the training pipeline of a language model.
Each episode adds a page here.

## The model at a glance

| Property | Value |
| --- | --- |
| Parameters | 14,585,856 |
| Vocabulary | 10,000 tokens |
| Context window | 256 tokens |
| Transformer blocks | 6 |
| Attention heads | 6 per block, 64 dimensions each |
| Embedding size | 384 |
| MLP hidden size | 1,536 (4 x 384) |
| Activation | GELU |
| Normalisation | LayerNorm before each sublayer (pre-norm) |
| Positions | Learned embeddings |
| Output head | Tied to the token embeddings |

All of these come from one dataclass, `GPTConfig`, in
[`model.py`](https://github.com/nenuadrian/mvGPT/blob/main/model.py). A trained model only loads
into a GPT built with the config it was trained with, since the config fixes every tensor's shape.

## Run it

```bash
git clone https://github.com/nenuadrian/mvGPT
cd mvGPT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python summary.py
python data.py
python pretrain.py 10
python sample.py checkpoints/pretrain/model.pt "Once upon a time"
python sft.py 10
python sample.py checkpoints/sft/model.pt "Words: dog, ball, happy. Story:"
tensorboard --logdir runs
```

`summary.py` builds the model and prints it the way PyTorch sees it: the modules, the parameters
by layer, the shape of the data after each layer, and a few tokens generated before any training.
`data.py` downloads TinyStories and tokenizes it, `pretrain.py` trains the model for ten minutes, and
`sample.py` writes stories with it. `sft.py` then fine-tunes it to write the story an instruction asks for.
Both training runs log to TensorBoard, under `runs/`.

## Pages

- [Architecture](architecture.md): the transformer, layer by layer, with the data flow and where
  the parameters are.
- [Pretraining](pretraining.md): the tokenizer, the data, next-token prediction and the training
  loop, sampling, and how we measure a model.
- [Supervised fine-tuning](sft.md): instructions and stories, a loss on the story alone, and how
  well the model follows instructions before and after.
- [Watching training](tracking.md): the runs in TensorBoard, and reading their logs in Python to
  plot them with matplotlib.
- [The KV cache](kv-cache.md): keeping the keys and values of earlier tokens, so each new token runs
  through the model on its own, and how much faster it writes on a CPU and a GPU.
- [Sharing and serving](deploy.md): the model on the Hugging Face Hub as a GPT-2, a version per
  episode, a Gradio app in Docker on a server, and the model in the browser with Transformers.js.
- [Profiling](profiling.md): the model as a graph in Model Explorer, PyTorch's profiler on the CPU
  and the GPU, and decisions in the code, measured.
- [Preferences, with DPO](dpo.md): pairs of the model's stories, scored by the words they use, and
  direct preference optimisation, with what it gains and what it costs.
- [Reinforcement learning, with PPO](ppo.md): the model writes, a reward scores it, and PPO's
  clipped objective, with a critic and a KL penalty, improves it, next to DPO.
- [LoRA](lora.md): fine-tuning a low-rank update beside frozen weights, with what it saves and what
  it learns, next to full fine-tuning.
- [Rotary position embeddings](rope.md): positions as rotations of the queries and keys, a cache
  that slides past the window, and RoPE next to learned positions.
- [Flash attention](flash-attention.md): attention without its table of scores, in PyTorch on the
  Mac, then as a Triton kernel on an NVIDIA GPU, forwards and backwards, next to PyTorch's own.
- [Serving many at once](serving.md): the model as an OpenAI-style completions API, the requests
  waiting written together as a batch, measured under a crowd of players, in Docker, and vLLM.
- [Text diffusion](diffusion.md): the same transformer, attending both ways, trained to fill in
  hidden tokens and writing a story by unhiding it, next to next-token prediction and LLaDA.
- [Reinforcement learning, with GRPO](grpo.md): eight stories per instruction, each scored against
  its group in place of PPO's critic, as DeepSeekMath and DeepSeek-R1 train, next to PPO.
- [A modern transformer block](modern-block.md): RMSNorm, SwiGLU, QK-norm and no biases, as options,
  from the papers that introduced them, with the parameters counted and tests against PyTorch's.
