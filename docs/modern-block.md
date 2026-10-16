# A modern transformer block

mvGPT's block follows GPT-2's: a LayerNorm before attention and another before the MLP, GELU in the
MLP, and a bias in each of the block's linear layers and LayerNorms. Recent open models change that
block in four ways, and mvGPT has each of them as an option:

- **RMSNorm** in place of LayerNorm: each token's channels are divided by their root mean square,
  with no mean subtracted and no bias.
- **SwiGLU** in place of GELU's MLP: two projections go up, one through SiLU, and their product
  comes back down.
- **QK-norm**: an RMSNorm on each head's queries and keys, before their dot product.
- **No biases**, in the linear layers or the norms.

Two of them come together in [LLaMA](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023),
whose section 2.2 lists pre-normalisation with RMSNorm, SwiGLU and [rotary position
embeddings](rope.md). The biases went in [PaLM](https://arxiv.org/abs/2204.02311) (Chowdhery et al.,
2022), and the QK-norm that recent models cite is from
[ViT-22B](https://arxiv.org/abs/2302.05442) (Dehghani et al., 2023). The
[Qwen3 report](https://arxiv.org/abs/2505.09388) (2025) describes all of them in one paragraph:
SwiGLU, RoPE, RMSNorm before each sublayer, no biases on the query, key and value projections, and
QK-norm, for stable training.

## RMSNorm

A LayerNorm subtracts the mean of a token's $n$ channels, divides by their standard deviation,
then applies a gain and a bias. RMSNorm, from [Zhang and Sennrich,
2019](https://arxiv.org/abs/1910.07467), skips the mean and the bias:

$$
\text{LayerNorm}(x)_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\, g_i + b_i
\qquad
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum_{j=1}^{n} x_j^2 + \epsilon}}\, g_i
$$

where $\mu$ and $\sigma^2$ are the channels' mean and variance. When the mean is zero, the two divide
by equal amounts. The paper's hypothesis is that LayerNorm's re-centring isn't what makes it work,
and in its experiments RMSNorm did about as well as LayerNorm in 7% to 64% less time, depending on
the model. It has half the parameters: a gain per channel, which starts at one.

`RMSNorm` in [`model.py`](https://github.com/nenuadrian/mvGPT/blob/main/model.py) is written out, as
GELU is: `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight`, with $\epsilon = 10^{-6}$.
It computes in float32 and casts back, as squaring in bfloat16, which `pretrain_ddp.py` trains in on
NVIDIA GPUs, loses precision. PyTorch has it as `nn.RMSNorm` and `F.rms_norm`, which the tests
compare it with. `norm_layer(config, size)` builds RMSNorm or LayerNorm, for the norms before
attention and the MLP, and the final one.

## SwiGLU

A gated linear unit multiplies two projections of its input, element by element, one of them passed
through a function first. [Shazeer, 2020](https://arxiv.org/abs/2002.05202) tried several functions
in the transformer's feed-forward layers, the MLP. SwiGLU uses Swish with $\beta = 1$, which PyTorch
calls SiLU, $x\,\sigma(x)$:

$$
\text{MLP}(x) = \text{GELU}(x W_\text{up})\, W_\text{down}
\qquad\longrightarrow\qquad
\text{MLP}(x) = \big(\text{SiLU}(x W_\text{gate}) \odot x W_\text{up}\big)\, W_\text{down}
$$

The SiLU side works as a gate: for each channel of each token, it scales the up projection by an
amount computed from the input, with weights of its own. With three matrices instead of two, the
hidden layer is two thirds as wide, $\frac{8}{3}d$ in place of $4d$, rounded up to a multiple of 64:
1,024 for mvGPT's 384, so
$3 \times 384 \times 1{,}024 = 2 \times 384 \times 1{,}536 = 1{,}179{,}648$ weights either way. In the
paper's Table 1, after 524,288 steps, SwiGLU's held-out log-perplexity is 1.636, against 1.679 for
GELU and 1.677 for ReLU, and GEGLU, with GELU as the gate, is a little lower, at 1.633. The paper
offers no explanation for why, and attributes their success "to divine benevolence".

`silu` is written out next to `gelu`, and the MLP gets a `gate_proj` next to `up_proj` and
`down_proj`, the names Hugging Face's Qwen3 code uses.

## QK-norm

Attention scores a query against a key as $q k^\top / \sqrt{d_k}$, with $d_k = 64$. Nothing bounds
that score: if the query and key projections grow, the score grows with the square of their size.
[Dehghani et al., 2023](https://arxiv.org/abs/2302.05442) saw their vision transformers of about 8
billion parameters diverge after a few thousand steps, with attention logits past 50,000 and
attention weights almost one-hot, and fixed it by normalising the queries and keys before the dot
product. [Wortsman et al., 2023](https://arxiv.org/abs/2309.14322) found the same instability in
small models trained at high learning rates (9.4 million parameters, at a learning rate of 0.1), and
the same fix working there. An earlier form, [Henry et al., 2020](https://arxiv.org/abs/2010.04245),
normalised queries and keys to unit length, so each score is a cosine, and scaled them by a learned
parameter instead of $1/\sqrt{d_k}$.

mvGPT normalises with RMSNorm, as [OLMo 2](https://arxiv.org/abs/2501.00656) does, and keeps
$1/\sqrt{d_k}$:

$$
\text{softmax}\left(\frac{\text{RMSNorm}(q)\;\text{RMSNorm}(k)^\top}{\sqrt{d_k}}\right) v
$$

With its gains at one, an RMSNorm gives each head's query and key a length of $\sqrt{64} = 8$, so a
score lies between $-8$ and $8$, however large the projections' weights. The learned gains can
still sharpen attention. `q_norm` and `k_norm` act on each head's 64 channels (their gains shared by
the heads), after the queries and keys are split into heads and before rotary embeddings turn them,
as in Hugging Face's Qwen3 code. A rotation keeps a vector's length, so the bound holds after it,
and the KV cache keeps keys that are normalised and rotated.

## No biases

In PaLM, "No biases were used in any of the dense kernels or layer norms", which its authors found
made training more stable for large models. ViT-22B dropped the biases of the query, key and value
projections and of its LayerNorms, for 3% better use of its accelerators, but kept those of the MLP,
which they found better for quality. `bias` has been a field of `GPTConfig` since the first episode,
true by default.

## In the code

Each change is a field of `GPTConfig`, so a checkpoint saved before them loads as the original
block:

| Field | Default | What it does |
| --- | --- | --- |
| `rmsnorm` | `False` | RMSNorm in place of each LayerNorm |
| `swiglu` | `False` | the gated MLP, $\frac{8}{3}d$ wide, rounded up to a multiple of 64 |
| `qk_norm` | `False` | RMSNorms on each head's queries and keys |
| `bias` | `True` | biases in the linear layers and the norms |
| `rope` | `False` | [rotary position embeddings](rope.md) |

`--modern` on `pretrain.py`, `pretrain_ddp.py` and `summary.py` turns on `rmsnorm`, `swiglu` and
`qk_norm`, and turns off `bias`; rotary embeddings stay their own flag, `--rope`.

```bash
python summary.py --rope --modern           # the modern block's modules and parameters
python pretrain.py 10 --rope --modern       # ten minutes of pretraining with it, to checkpoints/pretrain-rope-modern
python -m pytest tests/test_block.py -v
```

`pretrain_ddp.py --modern` saves to `checkpoints/pretrain-ddp-modern`, unless `--name` says
otherwise, so it doesn't overwrite `pretrain-ddp`.

[`tests/test_block.py`](https://github.com/nenuadrian/mvGPT/blob/main/tests/test_block.py) checks
that:

- `RMSNorm` matches `F.rms_norm`, and gives each token's channels a root mean square of one;
- without biases, the SwiGLU MLP and the GELU MLP have as many parameters;
- with QK-norm, scaling the query and key projections' weights by ten leaves attention's output
  unchanged, and without it, changes it;
- the modern model, with rotary embeddings, gives the logits of the whole sequence when it runs a
  token at a time with the KV cache;
- a config saved before these fields existed builds the original block, with its 14,585,856
  parameters, and the modern block has 24,960 fewer, with rotary embeddings.

## The parameter count

The changes leave the model about as large as it was. Per block:

| | GPT-2's block | Modern block |
| --- | --- | --- |
| Norms | $2 \times (384 + 384) = 1{,}536$ | $2 \times 384 = 768$ |
| Attention | $4 \times (384^2 + 384) = 591{,}360$ | $4 \times 384^2 = 589{,}824$ |
| QK-norm | | $2 \times 64 = 128$ |
| MLP | $2 \times 384 \times 1{,}536 + 1{,}536 + 384 = 1{,}181{,}568$ | $3 \times 384 \times 1{,}024 = 1{,}179{,}648$ |
| A block | 1,774,464 | 1,770,368 |

A modern block has 4,096 fewer parameters: 4,224 biases go, and QK-norm adds 128 gains. For the
whole model, the token embeddings (10,000 × 384, shared with the head), six blocks and the final norm:

| | GPT-2's block | Modern block |
| --- | --- | --- |
| With rotary embeddings | 14,487,552 | 14,462,592 |
| With learned positions (+98,304) | 14,585,856 | 14,560,896 |

That's 24,960 fewer, about 0.17% of the model.

## What's left out

The papers' reasons were measured on their own models: RMSNorm's speed on their GPUs, SwiGLU's
loss in Shazeer's T5 setup, QK-norm against the divergence of an 8-billion-parameter model.
Whether each change helps a model of mvGPT's size is a question for a training run. Recent models
also change more than this: Qwen3 has grouped query attention, PaLM runs attention and the MLP in
parallel, and OLMo 2 normalises the outputs of attention and the MLP instead of their inputs. Andrej Karpathy's
[nanochat](https://github.com/karpathy/nanochat) makes some of these choices differently: QK-norm,
an RMSNorm with no learnable parameters and no biases, but ReLU squared in a $4d$ MLP in place of
SwiGLU.
