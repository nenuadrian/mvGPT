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
```

## Docs

```bash
pip install -r requirements-docs.txt
mkdocs serve        # http://127.0.0.1:8000
```
