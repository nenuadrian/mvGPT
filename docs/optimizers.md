# Optimizers

All of mvGPT's training scripts use PyTorch's AdamW. This page writes AdamW from scratch, then races it
against three newer optimizers, Lion, Sophia and Muon, on the stories model, with a race set up to be
fair to each of them: one model, one order of batches, 1,000 steps, and a sweep of the learning rate for
each optimizer.

## AdamW

After a backward pass, each weight $w$ has a gradient $g$. Adam
([Kingma and Ba, 2014](https://arxiv.org/abs/1412.6980)) keeps two running averages for each weight,
the momentum $m$ and the average of the gradient squared $v$, and moves the weight by the first over the
square root of the second, so its step is about the learning rate $\eta$ in size whatever the scale of
its gradients. AdamW ([Loshchilov and Hutter, 2017](https://arxiv.org/abs/1711.05101)) shrinks the weights
apart from that update, so the decay isn't divided by $\sqrt{v}$ along with the gradient:

$$m \leftarrow \beta_1 m + (1 - \beta_1) g, \qquad v \leftarrow \beta_2 v + (1 - \beta_2) g^2$$

$$w \leftarrow w - \eta \frac{\hat m}{\sqrt{\hat v} + \epsilon} - \eta \lambda w$$

where $\hat m = m / (1 - \beta_1^t)$ and $\hat v = v / (1 - \beta_2^t)$ correct for both averages starting
at zero. On the first step, the update is $g / |g|$: each weight moves by the learning rate.

```python
m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
v.mul_(beta2).add_(p.grad.square(), alpha=1 - beta2)
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
p.mul_(1 - group["lr"] * group["weight_decay"])
p.sub_(group["lr"] * m_hat / (v_hat.sqrt() + group["eps"]))
```

`optimizers.py` has it as a `torch.optim.Optimizer`, with the settings mvGPT trains with: betas of 0.9
and 0.95, $\epsilon = 10^{-8}$ and a weight decay of 0.1. `tests/test_optimizers.py` trains two copies of a
small model, one with ours and one with `torch.optim.AdamW`, and checks that their weights match after
five steps.

## A fair race

`bakeoff.py NAME LR...` trains the stories model (rotary embeddings, the BPE tokenizer, no dropout) with
an optimizer, at each peak learning rate given:

- 1,000 steps of 48 windows of 256 tokens, about 12.3 million tokens, with a warmup of 100 steps and a
  cosine down to a tenth of the peak, as in pretraining, and the gradient clipped at 1.
- A seed fixes the starting weights and the batches, so with one seed, each optimizer starts from one
  model and reads the stories in one order. Sophia's labels come from a generator of their own.
- The held-out loss is measured each 100 steps on 40 held-out batches, 491,520 tokens, one set for all
  the runs.
- Each step is timed, with a sync before and after, and the optimizer's share of it: the optimizers'
  steps, and for Sophia, its estimate of the curvature.

Each run writes `results/bakeoff/NAME-LR-seedS.json`. The learning rate is the knob that matters most,
and the best rate of one optimizer can be far from another's, so each gets its own sweep: by factors of
two, then at $\sqrt 2$ on either side of its best. Its other settings come from its paper, or for
Sophia, from its code for GPT-2 small.

The held-out loss after 1,000 steps, with seed 0, the best of each in bold:

| Peak learning rate | AdamW | Lion | Sophia | Muon |
| --- | --- | --- | --- | --- |
| 1.5e-4 |  | 2.177 | 2.191 |  |
| 2.1e-4 |  | 2.101 | 2.107 |  |
| 3e-4 | 2.336 | **2.074** | **2.084** |  |
| 4.2e-4 |  | 2.100 | 2.128 |  |
| 6e-4 | 2.133 | 2.212 | 2.271 |  |
| 1.2e-3 | 1.998 |  | 2.949 |  |
| 1.7e-3 | 1.950 |  |  |  |
| 2.4e-3 | **1.934** |  |  | 1.769 |
| 3.4e-3 | 1.943 |  |  | 1.761 |
| 4.8e-3 | 2.050 |  |  | **1.758** |
| 6.8e-3 |  |  |  | 1.766 |
| 9.6e-3 |  |  |  | 1.786 |

AdamW at 6e-4, the learning rate `pretrain.py` uses, ends at 2.133, about 0.2 above AdamW at its best,
four times higher. Lion's and Sophia's best rates are lower than AdamW's, as their steps are often the
full learning rate, and at 1.2e-3 Sophia learns much more slowly. Muon's loss barely changes from 2.4e-3
to 9.6e-3.

## Lion

Lion ([Chen et al., 2023](https://arxiv.org/abs/2302.06675)) was found by a search over programs. It
keeps the momentum and nothing else, and moves each weight by the learning rate, up or down, by the sign
of a blend of the momentum and the gradient:

$$c = \beta_1 m + (1 - \beta_1) g, \qquad w \leftarrow w - \eta\,(\text{sign}(c) + \lambda w), \qquad m \leftarrow \beta_2 m + (1 - \beta_2) g$$

with the betas the paper used for language models, $\beta_1 = 0.95$ and $\beta_2 = 0.98$. Its search had
found 0.9 and 0.99, and with those, Lion's best here was 2.169, at 4.2e-4, about 0.1 worse. Its steps are
larger than most of Adam's, so its learning rate is 3 to 10 times smaller, and its weight decay 3 to 10
times larger: 1.0 here, ten times AdamW's.

## Sophia

Sophia ([Liu et al., 2023](https://arxiv.org/abs/2305.14342)) divides the momentum by an estimate of the
curvature $h$, the loss's second derivative for each weight, and clips each weight's step to the learning
rate:

$$w \leftarrow w - \eta\,\text{clip}\!\left(\frac{m}{\rho B h}, 1\right)$$

where $B$ is a batch's tokens. SophiaG, the version here, estimates $h$ once in 10 steps with the
Gauss-Newton-Bartlett estimator: labels drawn from the model's own predictions, and the squared gradient
of the loss on them, kept in a running average. The settings are those of its code for GPT-2 small: betas of
0.965 and 0.99, $\rho = 0.05$, and a weight decay of 0.2. `bakeoff.py` prints the share of weights whose step is
clipped, which Sophia's README tunes $\rho$ to keep between 50 and 90%.

## Muon

Muon ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/)) is for the weight matrices of
the hidden layers. It takes the momentum, Nesterov style, and orthogonalizes it: for $M = U \Sigma V^\top$,
the update is close to $U V^\top$, the directions of the momentum with their lengths set to 1. Five steps of
a Newton-Schulz iteration get there without a decomposition:

$$X \leftarrow a X + (b\, XX^\top + c\, (XX^\top)^2) X, \qquad (a, b, c) = (3.4445, -4.7750, 2.0315)$$

The update is scaled by $0.2 \sqrt{\max(A, B)}$ for an $A \times B$ matrix, Moonshot AI's scale from
[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982), which matches the size of AdamW's
updates, so the learning rate and the weight decay mean what they do for AdamW. The embedding (also the
head), the biases and the norms' gains stay with AdamW, at Muon's learning rate. The tests check that
`orthogonalize` keeps a matrix's singular vectors and brings its singular values between 0.6 and 1.2, and
that our Muon steps where `torch.optim.Muon` does (with `adjust_lr_fn="match_rms_adamw"`), to within its
bfloat16 rounding.

## The results

Each optimizer at its best rate, with seeds 0, 1 and 2. AdamW also ran at its best rate for 500,
750, 1,500 and 2,000 steps, each with its own warmup and cosine: 2.217, 2.042, 1.934 (1,000), 1.811
and 1.736. The 2,000-step run reads the stories about 1.2 times. "AdamW's steps" is how many steps
AdamW needs to reach an optimizer's loss with seed 0, read between the two runs around it, in log
steps. With seed 0, all the optimizers start from one model and read the batches in one order.

| Optimizer | Best rate | Seed 0 | Mean of 3 seeds | Spread | AdamW's steps | ms a step | In the optimizer | Kept per weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AdamW | 2.4e-3 | 1.934 | 1.933 | 0.003 | 1,000 | 462 | 11 | 2.00 |
| Lion | 3e-4 | 2.074 | 2.078 | 0.011 | 697 | 468 | 11 | 1.00 |
| Sophia | 3e-4 | 2.084 | 2.083 | 0.010 | 680 | 504 | 58 | 2.00 |
| Muon | 4.8e-3 | 1.758 | 1.760 | 0.005 | 1,840 | 540 | 72 | 1.23 |

- The seeds of each land within about 0.01 of each other, and the optimizers are 0.08 to 0.4 apart.
  Running one configuration again moves its loss by a few thousandths too: the Mac's GPU isn't fully
  deterministic.
- **Muon** reaches in 1,000 steps the loss AdamW needs about 1,840 for. Its Newton-Schulz iteration
  costs at most $Tm/B = 5 \times 384 / 12{,}288 \approx 16\%$ more work a step (Keller Jordan's runtime
  analysis), and about 72 ms of a step here, so in time it's still about 1.6 times as fast.
- **Sophia** and **Lion** are behind a tuned AdamW, worth about 680 and 700 of its steps. Sophia's
  share of clipped steps falls from 98% to about 87% by the end, so most of its weights take the sign of
  their momentum, like Lion's; its paper's runs are 100 times longer, with 10,000 estimates of the
  curvature to our 100.
- **AdamW at 6e-4** is worth about 610 steps of AdamW at 2.4e-3, so tuning the learning rate alone
  saved about 1.6 times the steps. Against that AdamW, Sophia and Lion would each look about 1.1 times
  as fast.
- The time a step takes is the median over all of an optimizer's runs; the whole steps move by tens of
  milliseconds with other work on the laptop, and the optimizer's part is what differs. Ours are plain
  Python loops over the weights; PyTorch's fused AdamW is faster.
- Memory: the stories model's 13.8 million weights take about 55 MB for each number kept per weight:
  110 MB for AdamW and Sophia, 68 MB for Muon (its momentum for the matrices, AdamW's two for the
  rest), 55 MB for Lion.

At scale, [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046)
(Wen et al., 2025) tuned ten optimizers on models of 0.1B to 1.2B parameters, with 1 to 8 times the
Chinchilla tokens: the speedups were lower than claimed, the fastest optimizers were the ones with
matrix preconditioners like Muon and Soap, and their speedup over AdamW fell from 1.3–1.4× at 0.1B
to 1.1× at 1.2B. The stories model is ten times smaller than their smallest, and its runs are far
shorter, under one token a weight, so its 1.8× is no guide to larger models.
[nanochat](https://github.com/karpathy/nanochat) trains its matrices with Muon and its embeddings, head
and scalars with AdamW, each group with a learning rate of its own.

```bash
python bakeoff.py adamw 1.2e-3 2.4e-3 4.8e-3   # a run for each peak learning rate, to results/bakeoff
python bakeoff.py muon 4.8e-3 --seed 1          # another seed
python bakeoff.py adamw 2.4e-3 --steps 2000     # another length
python bakeoff_report.py                        # each optimizer's sweep
python bakeoff_report.py --best                 # the best rate of each: seeds, AdamW's steps, time and memory
python -m pytest tests/test_optimizers.py
```
