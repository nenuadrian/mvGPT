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
```

`summary.py` builds the model and prints it the way PyTorch sees it: the modules, the parameters
by layer, the shape of the data after each layer, and a few tokens generated before any training.

## Pages

- [Architecture](architecture.md): the transformer, layer by layer, with the data flow and where
  the parameters are.
