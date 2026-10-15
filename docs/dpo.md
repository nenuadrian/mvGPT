# Preferences, with DPO

Supervised fine-tuning shows the model the answers we want. Preference optimisation shows it two
answers to one prompt and says which is better, which is often easier to give than a perfect
answer. Direct preference optimisation (DPO) learns from those pairs with a loss of its own, with no
reward model and no reinforcement learning.

## Preferences, scored by code

For a model like ChatGPT, people compare answers. Here, the preference is scored by code, as the
share of the words an instruction asks for that a story uses, with the matching `evaluate.py` uses
(a word counts in its plural, or with -d, -ed or -ing). [`preferences.py`](https://github.com/nenuadrian/mvGPT/blob/main/preferences.py)
has the fine-tuned model write four stories for each instruction that asks for words, at a
temperature of 1, and pairs the story that uses the most of the words, as the chosen one, with the
story that uses the fewest, as the rejected one. Ties are skipped, as they say nothing about which
is better.

```bash
python preferences.py 2000 200    # 2,000 training pairs and 200 held-out ones
```

Eight instructions with the same number of tokens are written for together, with each repeated four
times, as a batch of 32 stories that needs no padding. Held-out pairs come from validation
instructions past the 50 `evaluate.py` uses.

## The loss

DPO keeps a frozen copy of the fine-tuned model as the reference. For a pair, it compares how much
more likely the policy makes the chosen story than the reference does, with the same for the
rejected one:

$$
\text{loss} = -\log \sigma\Big(\beta \big[(\log \pi_\theta(y_w \mid x) - \log \pi_\text{ref}(y_w \mid x))
- (\log \pi_\theta(y_l \mid x) - \log \pi_\text{ref}(y_l \mid x))\big]\Big)
$$

$y_w$ is the chosen story, $y_l$ the rejected one, and each log-probability is the sum over the
story's tokens, with the instruction masked out as in SFT. The loss pushes the margin up: the chosen
story gains probability against the reference, and the rejected one loses it. $\beta$ (0.1) sets how
far the policy may move from the reference for a given margin, as the gains are measured against it.

```bash
python dpo.py 10    # ten minutes, from checkpoints/sft/model.pt to checkpoints/dpo/model.pt
python evaluate.py checkpoints/sft/model.pt checkpoints/dpo/model.pt
```

## What it does, and what it doesn't

With 2,000 pairs, where the chosen stories use about 60% of the words asked for and the rejected
ones about 16%, ten minutes of DPO (about 1,160 steps of 16 pairs) on an Apple M4 Pro gave:

| | Fine-tuned | DPO |
| --- | --- | --- |
| Requested words used | 27.3% | 63.3% |
| Stories that end within 200 tokens | 48% | 20% |
| Average story length | 178 tokens | 197 tokens |
| Loss on plain stories | 2.38 | 2.87 |
| Perplexity on plain stories | 10.8 | 17.7 |

Held-out pairs are ranked right about 63% of the time, while the training batches reach 100%:
2,000 pairs seen about nine times each are memorised. Measuring the checkpoints along the way, most
of the gain comes in the first 200 steps (about 52% of the words used, for a loss of 2.51), and the
rest of the run buys a few more points while the loss on plain stories keeps rising. Fewer steps, or
a larger $\beta$, keep the model closer to the reference.

The preference rewards using the words, not using them well. The stories use more of the words
asked for, but they're no better written, and some put the words where they don't fit. A reward
written in code gets what it measures, which is why preferences for assistants come from people, or
from a reward model trained on what people prefer.
