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
```

## Docs

```bash
pip install -r requirements-docs.txt
mkdocs serve        # http://127.0.0.1:8000
```
