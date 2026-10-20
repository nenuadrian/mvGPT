# A longer context

The [web model](web.md) was pretrained on windows of 1,024 tokens. With
[rotary embeddings](rope.md), it can read past its window, as nothing in it has a table of positions,
but its loss climbs when it does. This page stretches its rotations four ways, so it reads pages of
4,096 tokens, measures each, with no training and after fifteen minutes of fine-tuning, and works out
what the longer window costs.

## Past the window

Each of a head's 32 pairs of channels turns at its own speed, $\theta_i = 10000^{-2i/64}$ radians per
position, and a query's score against a key depends on $(n - m)\,\theta_i$, the distance between the
two tokens times each pair's speed. In a window of 1,024 tokens, pair 0 turns about 163 times, while
pair 31 turns about 7.8°. At 4,096 tokens apart, pair 31 is at about 31°, a difference no query and key
had in training, so the scores for those distances come from angles the model has no training for.

## Four ways to stretch

`GPTConfig` gets two fields: `rope_scaling`, one of `"pi"`, `"ntk"`, `"parts"` and `"yarn"`, and
`rope_factor`, by how much, so the window trained on is `block_size / rope_factor`. The defaults, `None`
and 1, leave the rotations as they were, so the checkpoints saved before load as they did.
`stretch(config, scaling, factor)` gives the config of a stretched model, with a window `factor` times
as long; the weights load into it as they are.

- **Position interpolation** ([Chen et al., 2023](https://arxiv.org/abs/2306.15595)) divides each
  speed by the factor, so position 4,096 turns as far as 1,024 did. The slow pairs stay within the
  angles of a window, and the fast ones lose resolution: on pair 0, neighbouring tokens are a quarter
  of a radian apart, where training had them a radian apart. The paper fine-tunes afterwards, within
  1,000 steps.
- **NTK-aware interpolation** (posted on Reddit by bloc97, 2023) raises the base from 10,000 to
  $10000 \times s^{64/62}$, about 41,829 for $s = 4$, which keeps pair 0's speed and divides pair 31's
  by $s$; the pairs between are slowed by less, so some slow pairs still turn past a window's angles.
- **NTK-by-parts** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071), section 3.2) counts each
  pair's turns in the window trained on, $r = L\theta_i / 2\pi$. Pairs with $r \geq \beta = 32$ keep
  their speeds, pairs with $r \leq \alpha = 1$ are divided by $s$, and those between are slowed on a ramp,
  $\gamma = (r - \alpha) / (\beta - \alpha)$: $\theta_i' = \gamma\,\theta_i + (1 - \gamma)\,\theta_i / s$.
  For mvGPT's heads of 64 channels and a window of 1,024 tokens, that's pairs 0–5 kept, 6–17 on the
  ramp and 18–31 divided by 4.
- **YaRN** (the same paper, section 3.3) is NTK-by-parts with a temperature $t$ on attention's scores,
  $\text{softmax}(q^\top k / (t\sqrt{d}))$, with $\sqrt{1/t} = 0.1 \ln s + 1$. The cosines and sines are
  multiplied by $\sqrt{1/t}$, about 1.14 for $s = 4$, so queries and keys are both longer, and the
  scores about 1.30 times larger.

```python
def rope_speeds(head_size, device, scaling=None, factor=1.0, trained=None):
    # A speed for each pair of channels, in radians per position: faster for the first pairs, slower for the last.
    speeds = 10_000 ** (-torch.arange(0, head_size, 2, device=device) / head_size)
    if scaling == "pi":
        return speeds / factor
    if scaling == "ntk":
        base = 10_000 * factor ** (head_size / (head_size - 2))
        return base ** (-torch.arange(0, head_size, 2, device=device) / head_size)
    if scaling in ("parts", "yarn"):
        turns = trained * speeds / (2 * math.pi)  # in the window trained on
        keep = ((turns - SLOW_TURNS) / (FAST_TURNS - SLOW_TURNS)).clamp(0, 1)
        return keep * speeds + (1 - keep) * speeds / factor
    return speeds
```

`tests/test_long.py` checks each method's speeds, the temperature, that stretching by 1 changes nothing,
and that a stretched model reads past its window, with the KV cache giving the logits of the whole
sequence.

## Loss by position

`long.py` measures the first 64 held-out web pages of more than 4,096 tokens, a loss for each
position, against a sliding window of the window trained on: windows a quarter of one apart, each
predicting its last quarter, so each token past the first window sees from 768 to 1,024 tokens. With no
training:

| Method | 0–1,023 | 1,024–2,047 | 2,048–3,071 | 3,072–4,095 |
| --- | --- | --- | --- | --- |
| A sliding window | 3.104 | 3.143 | 3.157 | 3.113 |
| No stretching, the whole page | 3.104 | 4.603 | 5.736 | 5.885 |
| Position interpolation | 4.410 | 4.561 | 4.628 | 4.647 |
| NTK-aware | 3.169 | 3.222 | 3.348 | 4.800 |
| NTK-by-parts | 3.320 | 3.456 | 3.510 | 3.498 |
| YaRN | 3.163 | 3.182 | 3.197 | 3.165 |

Without stretching, the loss climbs past the window. Position interpolation is high from the start, as
the fast pairs no longer tell nearby tokens apart. NTK-aware holds to about 2,500 positions, then rises.
NTK-by-parts and YaRN stay flat, and YaRN's temperature is worth about 0.3 of loss. Past the window,
YaRN is a little above the sliding window, and in the window, it costs about 0.06 against the model as
it was trained.

## The pass key

`passkey.py` runs the test from the position interpolation paper (its section 3.3), with its prompt:
a random five-digit key hidden in filler ("The grass is green. The sky is blue. ..."), asked for at the
end, "What is the pass key? The pass key is". Prompts of one to four windows, with the key at five
depths, from the start to right before the question, ten keys each, the same keys for each method. The
model answers greedily, and a key is found when its six tokens hold it. Percent found, by the prompt's
length:

| Method | 1,024 | 2,048 | 3,072 | 4,096 |
| --- | --- | --- | --- | --- |
| No stretching | 40% | 0% | 0% | 0% |
| Position interpolation | 0% | 0% | 0% | 0% |
| NTK-aware | 90% | 64% | 24% | 0% |
| NTK-by-parts | 96% | 96% | 88% | 70% |
| YaRN | 86% | 84% | 80% | 56% |
| Position interpolation, fine-tuned | 56% | 76% | 70% | 54% |
| YaRN, fine-tuned | 80% | 84% | 74% | 44% |

Most of the keys the stretched models miss are at the two ends: at the start of the longest prompts,
the furthest back, and right before the question, where the model writes words instead of the
number, such as "the key to the game." NTK-by-parts finds more keys than YaRN, though its loss is
higher. Without stretching, past the window, the model writes filler (". The green. The next").

A sliding window of 1,024 tokens can't see a key further back than that at all.

## Fifteen minutes at 4,096 tokens

`long_train.py METHOD MINUTES` stretches the web model by 4 and fine-tunes it on the 21,895 training
pages of more than 4,096 tokens: four pages a step, one at a time with their gradients added up, as two
at once don't fit in a laptop's memory, from random places in random long pages. AdamW at 5e-5, with no
weight decay, after a warmup of 10 steps, as the YaRN paper trains (its warmup was 20 steps, at 2e-5
for Llama 2). Each run
is fifteen minutes on an Apple M4 Pro, 127 steps and about 2.1 million tokens, at about 2,300 tokens a second, against the 3.6 billion the model was
pretrained on.

On the 64 pages of `long.py`, each fine-tuned model against a sliding window of its own weights:

| Model | In the window | Past it | Past it, with its own sliding window |
| --- | --- | --- | --- |
| The web model, a sliding window | 3.104 | 3.138 | |
| Position interpolation, no training | 4.410 | 4.612 | |
| Position interpolation, fine-tuned | 3.235 | 3.224 | 3.265 |
| YaRN, no training | 3.163 | 3.182 | |
| YaRN, fine-tuned | 3.134 | 3.112 | 3.162 |

On the 8 held-out pages each run evaluates, position interpolation's loss past the window goes from
4.70 to 3.30, still falling at the end, and YaRN's from 3.26 to 3.19 in the first 20 steps, where it
stays: YaRN needs less training to read the longer window, as its paper says. Fine-tuned, YaRN's loss
past the window is below the sliding window of the model it started from, so it uses the tokens far
back on ordinary web pages. Position interpolation uses them too, below its own sliding window, from
higher up. The pass key didn't improve with YaRN's fine-tuning: nothing in web pages asks for a number
from 4,000 tokens back.

## What it costs

The KV cache keeps $2 \times 12 \text{ blocks} \times 12 \text{ heads} \times 64 \times 4$ bytes =
73,728 bytes a token for the web model: 75 MB for a window of 1,024 tokens, 302 MB for 4,096. Attention's
scores grow with the square of the length: PyTorch's attention on MPS computes them all, so one block's
scores for a page of 4,096 tokens take $12 \times 4096^2 \times 4$ bytes, about 805 MB, against 50 MB
for 1,024, and training keeps them for each block until the backward pass. One page took about 20 GB
while fine-tuning, which is why `long_train.py` reads one page at a time. [Flash attention](flash-attention.md)
doesn't keep the whole matrix, which is how long contexts train on GPUs.

## At scale

[Qwen3](https://huggingface.co/Qwen/Qwen3-8B) supports 32,768 tokens natively and 131,072 with YaRN, a
`rope_scaling` of type `yarn` and factor 4 in its config; its model card notes that static YaRN can
lower the quality on shorter texts, and advises adding it when long contexts are needed.
[DeepSeek-V3](https://arxiv.org/abs/2412.19437) uses YaRN with $s = 40$, $\alpha = 1$, $\beta = 32$ and
$\sqrt{t} = 0.1 \ln s + 1$, and trains for 1,000 steps at 32K tokens, then 1,000 at 128K, from a
window of 4K.

```bash
python long.py checkpoints/pretrain-web/model.pt      # loss by position: a sliding window, and each method with no training
python passkey.py checkpoints/pretrain-web/model.pt   # the pass key, by length, for each method
python long_train.py yarn 15                          # fine-tune, stretched with yarn (or pi, ntk, parts), to checkpoints/long-yarn
python long.py checkpoints/long-yarn/model.pt         # a stretched model, as it is, against its own sliding window
python -m pytest tests/test_long.py
```
