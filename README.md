# mvGPT

A minimum viable GPT: a small decoder-only transformer, written from scratch in PyTorch, that you
can read from one end to the other. It has 14.6 million parameters and learns to write children's
stories from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), in minutes on a
laptop.

It's the code of the YouTube series [Minimum Viable GPT](https://mvgpt.nenuadrian.com), which builds
it on camera, one tagged step at a time (`ep001/step-1`, ...), and then takes it through the
training pipeline of a language model. The docs are at <https://nenuadrian.github.io/mvGPT/>.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python summary.py      # the model as PyTorch sees it: modules, parameters, shapes
python data.py         # download TinyStories, build the vocabulary, store the stories as token ids
python pretrain.py 10  # pretrain for 10 minutes
python sample.py checkpoints/pretrain/model.pt "Once upon a time"
python sft.py 10       # fine-tune on instructions for 10 minutes
python sample.py checkpoints/sft/model.pt "Words: dog, ball, happy. Story:"
python evaluate.py checkpoints/pretrain/model.pt checkpoints/sft/model.pt
tensorboard --logdir runs  # the training runs, at http://localhost:6006
python plot.py runs/*      # read the runs' logs and plot them to results/
python bench.py checkpoints/sft/model.pt --device cpu  # check the KV cache and time it
python hub.py checkpoints/sft/model.pt hf/sft  # save it as a GPT-2 for transformers, and as ONNX
python app.py              # write stories in a web app, at http://127.0.0.1:7860
python explore.py checkpoints/sft/model.pt --times     # the model as a graph, in Model Explorer
python profiler.py checkpoints/sft/model.pt --device cpu   # where the time goes
python decisions.py checkpoints/sft/model.pt --device cpu  # decisions in the code, measured
python preferences.py 2000 200   # pairs of stories, the one that uses more of the words preferred
python dpo.py 10                 # learn from the pairs with DPO for 10 minutes
python ppo.py 10 0.05            # reinforcement learning with PPO, a KL weight of 0.05, for 10 minutes
python lora.py 10                # fine-tune on instructions with LoRA, 2% of the weights, for 10 minutes
python pretrain.py 10 --rope     # pretrain with rotary position embeddings, to checkpoints/pretrain-rope
python context.py checkpoints/pretrain/model.pt checkpoints/pretrain-rope/model.pt  # past the window
python attention_bench.py --device cpu   # attention by hand and flash attention: time and memory, by length
python server.py         # the model as an OpenAI-style completions API, at http://127.0.0.1:8000
python load.py           # a crowd of players against the server: tokens a second, and how long they wait
python diffusion.py mask                         # a held-out story with some of its tokens hidden
python pretrain.py 10 --rope --bpe --diffusion   # a masked diffusion model, to checkpoints/pretrain-rope-bpe-diffusion
python diffusion.py write checkpoints/pretrain-rope-bpe-diffusion/model.pt --ladder 8 16 32 64  # write by unhiding
python diffusion.py fill checkpoints/pretrain-rope-bpe-diffusion/model.pt "Once upon a time, there was a little dog named Max." "Max was never scared of the dark again."
python bits.py checkpoints/pretrain-rope-bpe/model.pt checkpoints/pretrain-rope-bpe-diffusion/model.pt  # bits per character, a bound for diffusion
python diffusion_bench.py checkpoints/pretrain-rope-bpe-diffusion/model.pt --against checkpoints/pretrain-rope-bpe/model.pt --judge checkpoints/pretrain-ddp/model.pt
python pretrain.py 5 --rope --bpe --diffusion --init checkpoints/pretrain-rope-bpe/model.pt  # from a next-token model
python grpo.py 10 0.05           # reinforcement learning with GRPO: 8 stories per instruction, no critic, for 10 minutes
python summary.py --rope --modern   # a modern block: RMSNorm, SwiGLU, QK-norm and no biases
python pretrain.py 10 --rope --modern   # pretrain with it, to checkpoints/pretrain-rope-modern
python -m pytest tests/test_block.py    # check the modern block against PyTorch and the original block
python summary.py --rope --modern --kv-heads 2   # grouped-query attention: 2 heads of keys and values, and the KV cache's size
python -m pytest tests/test_gqa.py      # check it against multi-head attention, and the cache against the whole sequence
python data.py --web 1   # FineWeb-Edu: merges learned on web pages, the held-out pages, and one file of tokens
torchrun --standalone --nproc_per_node 1 pretrain_ddp.py 470 --web --compile --block 1024 --batch 24 --accum 4  # a night on one GPU
python hellaswag.py checkpoints/pretrain-web/model.pt   # HellaSwag; gpt2 for GPT-2 small
python bits.py checkpoints/pretrain-ddp/model.pt checkpoints/pretrain-web/model.pt --web  # bits per character on web pages
```

## Docs

```bash
pip install -r requirements-docs.txt
mkdocs serve        # http://127.0.0.1:8000
```
