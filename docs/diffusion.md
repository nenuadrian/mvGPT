# Text diffusion

Every model so far writes a token at a time, left to right: next-token prediction, an
autoregressive model. A masked diffusion model writes the other way round. It starts from a window
of `[MASK]` tokens, and over a number of steps, unhides them, a few at a time, anywhere in the
window, each step seeing the whole story so far, on both sides of each gap.

It's the same transformer. What changes is the attention mask, the loss, and how it writes: the
recipe of [MDLM](https://arxiv.org/abs/2406.07524) and [LLaDA](https://arxiv.org/abs/2502.09992),
whose [code](https://github.com/ML-GSAI/LLaDA) shows it at the scale of an 8-billion-parameter
model trained from scratch on 2.3 trillion tokens. Its
[GUIDELINES.md](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md) has the loss in a dozen
lines, and its [generate.py](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) the sampler.

## Training: fill in what's hidden

A window of $T$ tokens, $x$, is noised by drawing a rate $t$ between 0 and 1 and replacing each token
by `[MASK]` with probability $t$. The model sees the noised window, $x_t$, and is scored on the
tokens it hid:

$$
\mathcal{L} = \mathbb{E}_{t,\,x_t}\left[\frac{1}{t}\sum_{i:\,x_t^i = \texttt{[MASK]}} -\log p_\theta(x^i \mid x_t)\right] \;\ge\; -\log p_\theta(x)
$$

Dividing by $t$ is what makes it a bound on the negative log likelihood, so a diffusion model's
loss can be compared with a next-token model's, as an upper bound. A rate drawn anywhere from 0
to 1, not BERT's fixed 15%, is what makes it a model that can write: at rates near 1, it learns to
start a story from almost nothing.

In the code:

- `GPTConfig(causal=False)`: attention without the mask, each token seeing every other, before and
  after it. It costs no parameters.
- `[MASK]` is one more token, after the tokenizer's ids: one row more in the embedding, 384
  parameters on the laptop's model.
- `diffusion_loss` in `diffusion.py` draws a rate per row, spread evenly across the batch from one
  random offset, and divides each row's loss on its hidden tokens by its rate, then by the window's
  length, which puts it per token, on the scale of next-token prediction's loss.
- The head still predicts each position's next token, as [Dream 7B](https://hkunlp.github.io/blog/2025/dream/)'s
  does: a `[MASK]` at position $i$ is filled from the output at position $i - 1$. So the first token
  of a window is never hidden, and a next-token model's weights start out close to the job (see
  below).

```bash
python diffusion.py mask                         # a held-out story with 10%, 50% and 90% hidden
python pretrain.py 10 --rope --bpe --diffusion   # to checkpoints/pretrain-rope-bpe-diffusion
```

## Writing: unhide the surest guesses

`denoise` starts from a canvas, the window's length, with end of text, then the prompt, then
`[MASK]`. Each step runs the model on the whole canvas, samples a guess for every hidden token at
once, and keeps the ones it was surest of, enough that the last step writes the rest; the others
stay hidden for a fresh guess, with more of the story around them. That's LLaDA's low-confidence
remasking. A story ends at the next end of text; the rest of the canvas is the start of another,
as windows in training run on into the next story.

```bash
python diffusion.py write checkpoints/pretrain-rope-bpe-diffusion/model.pt --ladder 8 16 32 64
python diffusion.py fill checkpoints/pretrain-rope-bpe-diffusion/model.pt "Once upon a time, there was a little dog named Max." "Max was never scared of the dark again."
```

`fill` gives the beginning of a story and its end, with a gap of `[MASK]` between them, which a
next-token model can't do directly: it would have to write the end without seeing it.

Guesses made in the same step don't see each other, so two neighbours can both become a likely word,
as in "play play". More steps, fewer tokens a step, fewer such clashes, and more passes of the model.

## On the laptop

Ten minutes of pretraining each, on an Apple M4 Pro, with rotary embeddings and the BPE tokenizer,
next to the next-token model from the [BPE](bpe.md) page:

| | Next-token | Diffusion |
| --- | --- | --- |
| Parameters | 13,793,280 | 13,793,664 (the `[MASK]` row) |
| Steps in 10 minutes | 1,234 | 1,216 |
| Held-out loss per token (`bits.py`'s windows) | 2.12 | ≤ 3.08 |
| Bits per character | 0.763 | ≤ 1.109 |

```bash
python bits.py checkpoints/pretrain-rope-bpe/model.pt checkpoints/pretrain-rope-bpe-diffusion/model.pt
```

`bits.py` also gives the diffusion model's loss per hidden token, by the share of tokens hidden:

| Hidden | 3% | 16% | 34% | 53% | 72% | 84% | 97% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Loss | 1.91 | 2.03 | 2.37 | 2.87 | 3.61 | 4.36 | 5.45 |

With a few tokens hidden, filling them in from both sides is easier than predicting the next token
from the ones before it (2.12). With nearly all of them hidden, there's little left to go on. The
bound averages over all the rates, so it pays for the hard end, which next-token prediction, reading
the story in a single order, doesn't have to learn.

`diffusion_bench.py` writes 16 stories from "Once upon a time" with each, one at a time, on the
Mac's GPU, and scores them with the 91-million-parameter next-token model from
[Training on several GPUs](distributed.md):

```bash
python diffusion_bench.py checkpoints/pretrain-rope-bpe-diffusion/model.pt --against checkpoints/pretrain-rope-bpe/model.pt --judge checkpoints/pretrain-ddp/model.pt
```

| | Steps | ms a story | Tokens | Judge's perplexity | Distinct 3-word runs |
| --- | --- | --- | --- | --- | --- |
| Held-out stories | | | 183 | 3.21 | 98% |
| Next-token, KV cache | | 432 | 163 | 6.38 | 96% |
| Diffusion | 16 | 118 | 175 | 25.2 | 85% |
| Diffusion | 32 | 234 | 184 | 18.4 | 79% |
| Diffusion | 64 | 465 | 174 | 11.4 | 73% |
| Diffusion | 128 | 960 | 175 | 10.1 | 74% |
| Diffusion | 256 | 1,884 | 172 | 7.17 | 71% |

Each step runs the model over the whole window, so the time grows with the steps; at about 64, it
takes about as long as writing a token at a time with the cache. With fewer steps, more tokens are
guessed at once, without seeing each other, and the judge scores the stories as less likely. The
judge likes repetition, though, and with more steps, the ten-minute diffusion model repeats itself
more: the share of three-word runs it doesn't repeat falls from 85% to 71%.

## Starting from a next-token model

Training a diffusion model from scratch is the expensive way. [Dream 7B](https://hkunlp.github.io/blog/2025/dream/)
started from the weights of Qwen2.5 7B, a next-token model. `from_autoregressive` in `diffusion.py`
loads a next-token model's weights into a diffusion model, with the new model's row for `[MASK]`
added to the embedding (and so to the head, which shares it). As the head still predicts the next
token, the weights start close to their new job: what's new to them is seeing the tokens after
each position, and the masks.

```bash
python pretrain.py 5 --rope --bpe --diffusion --init checkpoints/pretrain-rope-bpe/model.pt   # to checkpoints/pretrain-rope-bpe-diffusion-converted
torchrun --standalone --nproc_per_node 4 pretrain_ddp.py 15 --full --compile --batch 96 --diffusion --init checkpoints/pretrain-ddp/model.pt --name diffusion-ddp-converted
```

On the laptop, five minutes from the next-token model above:

| | Diffusion, from scratch | Diffusion, from the next-token model |
| --- | --- | --- |
| Training | 10 minutes | 10 minutes next-token, then 5 |
| Held-out loss at the first step | 9.12 | 5.00 |
| Held-out loss at the end | 3.11 | 2.80 |
| Bits per character | ≤ 1.109 | ≤ 1.009 |

## On four GPUs

Four RTX 4090s rented on [Vast.ai](https://cloud.vast.ai/?ref_id=746308) (a referral link), the kind of
machine the [next-token model](distributed.md) trained on, with its settings: GPT-2 small's shape,
512 tokens of context, 96 windows per GPU per step, `torch.compile`, and all of TinyStories.

```bash
ssh gpu 'cd mvGPT; nohup torchrun --standalone --nproc_per_node 4 pretrain_ddp.py 30 --full --compile --batch 96 --diffusion --name diffusion-ddp > diffusion.log 2>&1 < /dev/null &'
```

| | Next-token | Diffusion, from scratch | Diffusion, from the next-token model |
| --- | --- | --- | --- |
| Training | 30 minutes | 30 minutes | 30 minutes next-token, then 15 |
| Steps | 4,096 | 3,869 | 1,935 |
| Tokens trained on | 805 million | 761 million | 380 million |
| Tokens per second, evaluations included | 446,000 | 421,000 | 420,000 |
| Held-out loss per token (`bits.py`'s windows) | 1.07 | ≤ 1.64 | ≤ 1.63 |
| Bits per character | 0.386 | ≤ 0.588 | ≤ 0.585 |

Attention both ways computes all of the scores, where causal attention skips the blocks above the
diagonal, and the loss keeps a value per token, so a diffusion step takes about 5% longer.

With the same thirty minutes, diffusion's bound ends about 52% above next-token prediction, and
the next-token model was under diffusion's final held-out loss within its first 500 steps. The bound
is loose by an amount we can't measure, but the gap is what papers on scaling masked diffusion find:
[SMDM](https://arxiv.org/abs/2410.18514), from LLaDA's authors, needed about sixteen times the
pretraining for a masked diffusion model to write as well as a next-token one.

Converted from the next-token model, the diffusion loss starts at 5.62 (9.15 from scratch), is at
2.03 after 250 steps (2.82 from scratch), and ends at 1.62 after 15 minutes, a little under thirty
minutes from scratch. It had the next-token model's thirty minutes first, but when a next-token
model already exists, converting it halves the time to a diffusion model this good.

The loss per hidden token by the share hidden, for the model from scratch:

| Hidden | 3% | 16% | 34% | 53% | 72% | 84% | 97% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Loss | 0.38 | 0.48 | 0.71 | 1.16 | 2.07 | 3.19 | 5.20 |

With a few tokens hidden, filling them in is much easier than predicting the next token (1.07 per
token); with nearly all of them hidden, it's close to writing from nothing. The bound averages over
all of these, which next-token prediction doesn't have to learn.

### Writing on one RTX 4090

`diffusion_bench.py` with the converted model, eight stories each, one at a time, scored by the
next-token model (which flatters its own stories, so the held-out stories are the reference):

| | Steps | ms a story | Tokens | Judge's perplexity | Distinct 3-word runs |
| --- | --- | --- | --- | --- | --- |
| Held-out stories | | | 199 | 3.35 | 97% |
| Next-token, KV cache | | 1,111 | 172 | 2.03 | 97% |
| Diffusion | 16 | 122 | 414 | 23.2 | 71% |
| Diffusion | 32 | 238 | 319 | 11.2 | 77% |
| Diffusion | 64 | 475 | 158 | 7.75 | 97% |
| Diffusion | 128 | 950 | 146 | 5.13 | 97% |
| Diffusion | 256 | 1,901 | 185 | 3.14 | 96% |

Each step runs the whole window of 512, so the time grows with the steps. With few steps, many
tokens are guessed at once without seeing each other, and the stories repeat themselves and often
don't reach an end of text, running to the end of the window. At 256 steps, about two tokens a
step, the judge finds them as likely as real stories, in about 1.7 times the next-token model's time.
That's what LLaDA's [FAQ](https://github.com/ML-GSAI/LLaDA#3-what-is-the-sampling-efficiency-of-llada)
says too: it samples slower than an autoregressive model, as it writes a fixed length, can't use a
KV cache, and does best with as many steps as tokens.

The machine, rented for about an hour, cost $2.90.

## What's real, and what's scaled down

- The objective, the bound and the sampler are the ones LLaDA uses at 8 billion parameters.
- LLaDA samples with argmax (temperature 0), in blocks from left to right ("semi-autoregressive"),
  and uses classifier-free guidance; ours samples with a temperature and top k, like our
  next-token model's `generate`.
- LLaDA trains 1% of its windows at random lengths, so it doesn't always expect a full window; ours
  always writes a full window and cuts the story at end of text.
- There's no KV cache: each step runs the whole canvas, both ways.
