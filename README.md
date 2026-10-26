# mvGPT

A minimum viable GPT: a small decoder-only transformer, written from scratch in PyTorch, that you
can read from one end to the other. It has 14.6 million parameters and learns to write children's
stories from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), in minutes on a
laptop.

It's the code of the YouTube series [Minimum Viable GPT](https://mvgpt.nenuadrian.com), which builds
it on camera, one tagged step at a time (`ep001/step-1`, ...), and then takes it through the
training pipeline of a language model. The docs are at <https://nenuadrian.github.io/mvGPT/>.

<!-- notebooks -->
## Each episode in Colab

A notebook for each episode runs its commands on its code, as the episode left it (its
`epNNN/end` tag), on a Colab GPU. Models from earlier episodes come from
[the Hub](https://huggingface.co/nenuadrian/mvGPT-checkpoints), the files the videos used.

| | Episode | |
|---|---|---|
| 1 | [Build a GPT from scratch in PyTorch](https://youtu.be/Vff1s2GzI_o) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep001-architecture.ipynb) |
| 2 | [Pretraining a GPT from scratch in PyTorch](https://youtu.be/KA0aVTQ5Wak) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep002-pretraining.ipynb) |
| 3 | [SFT: fine-tuning a GPT to follow instructions](https://youtu.be/WUsVr66Bgq0) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep003-sft.ipynb) |
| 4 | [KV cache from scratch in PyTorch](https://youtu.be/JPlzOZouQxQ) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep004-kv-cache.ipynb) |
| 5 | [Deploy a GPT: Hugging Face, Gradio, a browser](https://youtu.be/rn72Y_9dVE0) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep005-deploy.ipynb) |
| 6 | [Profiling a GPT with PyTorch's profiler](https://youtu.be/7C0USj9KtI0) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep006-profiling.ipynb) |
| 7 | [DPO: teaching a GPT from preferences](https://youtu.be/aQ4WxQfB2S4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep007-dpo.ipynb) |
| 8 | [PPO: reinforcement learning for a GPT](https://youtu.be/8ewa70rUPME) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep008-ppo.ipynb) |
| 9 | [LoRA: fine-tuning a GPT with 2% of its weights](https://youtu.be/m5THIkAJJFk) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep009-lora.ipynb) |
| 10 | [RoPE: rotary position embeddings from scratch](https://youtu.be/CVeEPW67Gv8) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep010-rope.ipynb) |
| 11 | [BPE in C++: a byte-level tokenizer for a GPT](https://youtu.be/_o0mJYWdKQQ) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep011-bpe.ipynb) |
| 12 | [Distributed training in PyTorch: 4 GPUs, DDP](https://youtu.be/5-RJVvuGZyE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep012-distributed.ipynb) |
| 13 | [Flash attention from scratch: PyTorch, Triton](https://youtu.be/HEYzYWquKos) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep013-flash-attention.ipynb) |
| 14 | [Batching: serving a GPT to many users at once](https://youtu.be/6XGhd_83uwU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep014-serving.ipynb) |
| 15 | [Text diffusion: a GPT that writes all at once](https://youtu.be/LVWHstzgqE4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep015-diffusion.ipynb) |
| 16 | [GRPO: reinforcement learning without a critic](https://youtu.be/SoFl6r8Qz-0) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep016-grpo.ipynb) |
| 17 | [RMSNorm, SwiGLU, QK-norm: a modern GPT block](https://youtu.be/cPiQO4AilGY) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep017-modern-block.ipynb) |
| 18 | [Grouped-query attention: a smaller KV cache](https://youtu.be/SpKMenz36TM) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep018-gqa.ipynb) |
| 19 | [Pretraining on the web: a GPT overnight](https://youtu.be/15vafg1wdpI) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep019-web.ipynb) |
| 20 | [LLM tool calling from scratch: a calculator](https://youtu.be/UrlqjWEuYH4) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep020-tools.ipynb) |
| 21 | [AI agent from scratch, with subagents](https://youtu.be/9SYXGDioA6g) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep021-agent.ipynb) |
| 22 | [Long context LLM: stretching RoPE with YaRN](https://youtu.be/2C2JYv2agzo) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep022-long-context.ipynb) |
| 23 | [Muon vs AdamW, Lion and Sophia: a fair race](https://youtu.be/0ysdFCrmdlU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep023-optimizers.ipynb) |
| 24 | [Text to Cypher: our GPT asks a graph database](https://youtu.be/NxeRk1Q8LXY) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nenuadrian/mvGPT/blob/main/notebooks/ep024-knowledge-graph.ipynb) |
<!-- /notebooks -->

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
python tool_data.py      # 50,000 questions for the calculator, and five held-out sets
python tool_sft.py 10    # fine-tune the web model to call it, for 10 minutes
python tool_eval.py checkpoints/tools/model.pt   # the calls, right and with the calculator off; --shots for a pretrained model
python tools.py checkpoints/tools/model.pt "What is 48213 * 907?"
python -m pytest tests/test_tools.py   # the calculator, what it refuses, and the loop
python agent_data.py     # 60,000 traces for the agent and its subagents
python agent_sft.py 30   # fine-tune the web model into all three roles, for 30 minutes
python agent_eval.py checkpoints/agent/model.pt   # held-out tasks, and subagents against one context
python agent.py checkpoints/agent/model.pt "Add up the numbers in jan.txt and feb.txt."   # in ./sandbox
python agent_app.py      # the agent in a chat, at http://127.0.0.1:7861
python -m pytest tests/test_agent.py   # the sandbox, the roles, and a subagent's context
python long.py checkpoints/pretrain-web/model.pt      # loss by position past the window: a sliding window, and four ways to stretch RoPE
python passkey.py checkpoints/pretrain-web/model.pt   # a pass key hidden in filler, by length, for each way
python long_train.py yarn 15                          # fine-tune at 4,096 tokens, stretched with yarn (or pi), for 15 minutes
python -m pytest tests/test_long.py   # each way's speeds, and a stretched model against the whole sequence
python bakeoff.py adamw 1.2e-3 2.4e-3 4.8e-3   # train the stories model 1,000 steps at each peak learning rate (or lion, sophia, muon)
python bakeoff.py muon 4.8e-3 --seed 1          # another seed; --steps 2000 for another length
python bakeoff_report.py                        # each optimizer's sweep, by learning rate
python bakeoff_report.py --best                 # each at its best: the seeds, AdamW's steps, the time and the memory
python -m pytest tests/test_optimizers.py   # ours against PyTorch's AdamW and Muon, and what each update does
python graph_data.py ../smollerGraph/data/wikidata.txt   # 30,000 questions answered by smollerGraph's sg serve, and four held-out sets
python graph_sft.py 20   # fine-tune the agent to ask it in Cypher, for 20 minutes
python graph_eval.py checkpoints/graph/model.pt   # the queries, right and with the graph off
python graph.py checkpoints/graph/model.pt "What family is the Red fox in?"
python -m pytest tests/test_graph.py   # the answers as text, the quotes, the loop, and the questions
python graph_agent_data.py ../smollerGraph/data/wikidata.txt   # the agent's tasks that ask the graph, with episode 21's (run agent_data.py first)
python graph_agent_sft.py 30   # fine-tune episode 24's model into the agent with a graph role, for 30 minutes
python graph_agent_eval.py checkpoints/graph-agent/model.pt   # the new tasks, the graph role on its own, and episode 21's tasks
python agent.py checkpoints/graph-agent/model.pt "How much heavier is the Okapi than the Baikal Seal?"
python -m pytest tests/test_agent.py tests/test_graph_agent.py   # the graph role, and each task replayed through the agent
```

## Docs

```bash
pip install -r requirements-docs.txt
mkdocs serve        # http://127.0.0.1:8000
```
