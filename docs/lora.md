# LoRA

Fine-tuning changes every weight of the model, so each fine-tuned version is a full copy of it, and
training keeps optimiser state for every weight. Low-rank adaptation (LoRA), from Microsoft's paper
of 2021, freezes the pretrained weights and learns a small update for each of some layers, as the
product of two thin matrices:

$$
h = W x + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times d_\text{in}},\; B \in \mathbb{R}^{d_\text{out} \times r}
$$

with a rank $r$ much smaller than the layer's size. $B$ starts at zero, so the adapted model starts
as the pretrained one, and $\alpha / r$ scales the update.

## In the code

[`lora.py`](https://github.com/nenuadrian/mvGPT/blob/main/lora.py) wraps each linear layer of the
attention and the MLP, in each block, in a `LoRALinear`: the frozen layer, plus $A$ and $B$. It then
fine-tunes on instructions, as `sft.py` does, from the pretrained model, training $A$ and $B$ alone.
At the end, it saves the adapters on their own, and merges them into the weights, $W + \frac{\alpha}{r} B A$,
so the merged model runs as fast as the original, and loads like any other checkpoint.

```bash
python lora.py 10    # ten minutes, to checkpoints/lora: adapters.pt, and model.pt, merged
python evaluate.py checkpoints/pretrain/model.pt checkpoints/sft/model.pt checkpoints/lora/model.pt
```

## What it costs, and what it learns

Ten minutes each, from the pretrained model, on an Apple M4 Pro, measured by `evaluate.py`:

| | Pretrained | Fine-tuned, all weights | LoRA, rank 8 |
| --- | --- | --- | --- |
| Weights trained | | 14.6 million | 331,776 (2.2%) |
| What's saved | | 58.5 MB | 1.35 MB of adapters |
| AdamW's state | | 119 MB | 2.7 MB |
| Loss on instructions' stories | 2.60 | 2.13 | 2.40 |
| Requested words used | 2.7% | 27.3% | 10.0% |
| Loss on plain stories | 2.25 | 2.38 | 2.35 |

LoRA trains 2.2% of the weights, saves a file 43 times smaller, and keeps optimiser state 44 times
smaller, at about the speed per step of full fine-tuning, as the gradients still flow back through
the frozen layers. It learns less in ten minutes: its held-out loss levels off at about 2.27, where
full fine-tuning's reaches 2.02, and rank 32 does little better (2.24). And it forgets less of
pretraining, with a lower loss on plain stories.

At this size, full fine-tuning is cheap, so LoRA's savings matter less than they would for a model
of billions of parameters, where the saved file and the optimiser's state are gigabytes, and one
frozen copy of the model can serve many adapters, one per task or customer.
