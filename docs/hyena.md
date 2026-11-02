# Hyena: a GPT without attention

[Hyena](https://arxiv.org/abs/2302.10866) (Poli, Massaroli et al., 2023) replaces attention in the block with long
convolutions, as long as the window, between gates read off the tokens. A convolution over the whole window is
computed with the fast Fourier transform in $O(T \log T)$ operations, where attention's table of scores takes
$O(T^2)$. `hyena.py` writes the paper's operator of order 2 as its authors' code computes it, and `GPTConfig(mixer=
"hyena")` puts it in each block, or in each block but those listed in `attention_layers`, for a hybrid.

## Attention and Hyena, as matrices

For one head, attention is a matrix set by the input, times the values:

$$
y = A(u)\, v, \qquad A_{ts} = \frac{\exp(q_t \cdot k_s / 8)}{\sum_{r \le t} \exp(q_t \cdot k_r / 8)} \;\; (s \le t)
$$

The weight token $t$ puts on token $s$ comes from comparing them, query against key. The paper's operator of order
$N$ is a recurrence of convolutions and gates, starting from the values:

$$
z^1_t = v_t, \qquad z^{n+1}_t = x^n_t\, (h^n * z^n)_t, \qquad y_t = z^{N+1}_t, \qquad (h * z)_t = \sum_{s=0}^{t} h_{t-s}\, z_s
$$

Its authors' code, for order 2, makes $x^1$, $x^2$ and $v$ with one projection of $3C$ channels and a short
convolution of 3 taps on each channel (the first convolution of the recurrence), then one long filter $h$ between two
gates, with a skip $b$ for each channel:

$$
y_t = x^2_t \Big( \sum_{s \le t} h_{t-s}\, x^1_s\, v_s + b\, x^1_t\, v_t \Big)
\qquad\Longleftrightarrow\qquad
y = D_{x^2}\, (S_h + b I)\, D_{x^1}\, v
$$

$D_x$ has $x$ on its diagonal, and $S_h$, the convolution as a matrix, has $h_{t-s}$ at row $t$, column $s$: it's
constant along each diagonal, a Toeplitz matrix. So in Hyena, the weight token $t$ puts on token $s$ is
$x^2_t\, h_{t-s}\, x^1_s$: the filter at their distance, scaled by a gate at each end. Each of the $C$ channels has a
matrix of its own (`Hyena.matrices`), where the 64 channels of an attention head share their weights.

## The convolution, with the FFT

The convolution theorem turns a convolution into a product, frequency by frequency:
$h * z = \mathrm{IFFT}(\mathrm{FFT}(h) \odot \mathrm{FFT}(z))$. The FFT's convolution is circular, so `fft_conv` pads
both to $n = 2T$, and what wraps round lands past the $T$ outputs it keeps. For $T = 4{,}096$, the sums written out
(`direct_conv`) take 8,390,656 multiply-adds a channel; three transforms take about $3 \times \tfrac52 n \log_2 n
\approx$ 800,000 operations.

## The filter

The long filter is implicit, the paper's equation 7: a small network of the position, times a fade.

$$
h_t = e^{-\alpha_c\, t/(L-1)} \cdot \mathrm{FFN}(p_t), \qquad
p_t = \Big(\tfrac{t}{L-1},\; \cos\tfrac{2\pi k t}{L},\; -\sin\tfrac{2\pi k t}{L}\Big)_{k=0..15}
$$

The network is $33 \to 64 \to 64 \to 64 \to C$ with sine activations (frequency from 14, shared, as in the authors'
code), and $\alpha_c$ goes from $\ln 100 / 1.5 \approx 3.1$ to $\ln 100 / 0.3 \approx 15.4$ across the channels:
the fastest falls to 1% by 30% of the window, the slowest is at about 5% at its end. That's $10{,}560 + 65\,C$
weights for any window: 35,520 for 384 channels, where a filter learned tap by tap would need $C \times L$.

## Writing, one token at a time

A Hyena block keeps $z = x^1 v$ for each token in the cache, $C$ numbers, where attention keeps a key and a value,
$2C$. A new token's output is the sum of the cached $z$ weighted by the filter at each distance, $t\,C$ multiply-adds
after $t$ tokens, against attention's $2\,t\,C$: both grow with the text, and the FFT doesn't help a single token.
[Laughing Hyena](https://arxiv.org/abs/2310.18780) (Massaroli et al., 2023) distils the filters into a recurrence with
a state of a fixed size.

## What we measured

**One layer, forward and backward,** on 65,536 tokens a batch (`hyena_bench.py`). Attention's time per token grows
with the window, Hyena's barely does. On the Mac's GPU, Hyena is the faster from 1,024 tokens (6.8x at 8,192, on
16,384 tokens a batch). On an RTX 4090, PyTorch's attention is FlashAttention in bfloat16 on the tensor cores, while
our FFTs run in float32, and the two cross at about 16,384 tokens: Hyena takes 31.9 ms at 65,536 against attention's
80.4 ms, 2.5x faster, at 384 channels, and the same at 768. The authors' fused FFT convolution crossed FlashAttention
between 4K and 8K; [FlashFFTConv](https://arxiv.org/abs/2311.05908) (Fu et al., 2023) puts the FFT on the tensor cores.

**Six small models,** 6 blocks of 384 channels, half an hour each on a 4090, 65,536 tokens a step, on FineWeb-Edu
(`hyena_runs.sh`, `hyena_eval.py`):

| | attention | Hyena | hybrid (attention in block 3) |
|---|---|---|---|
| 1,024 tokens: tokens a second | 516K | 322K | 341K |
| 1,024: held-out loss after 30 minutes | 3.695 | 3.994 | 3.755 |
| 4,096: tokens a second | 437K | 318K | 333K |
| 4,096: held-out loss after 30 minutes | 3.734 | 4.065 | 3.784 |
| 4,096: loss in the first and last 256 tokens | 3.93, 3.70 | 4.12, 4.20 | 3.95, 3.75 |
| 1,024: random tokens twice, second half copied | 17% | 0% | 63% |
| 4,096: pass keys found | 22.5% | 0% | 16.5% |

At equal steps, so on as many tokens, the hybrid learns about as fast as attention and Hyena more slowly; Hyena's
loss rises across a window of 4,096, so it makes no use of the far end of it at this size. `flow.py` on "Marie Curie
was born in Warsaw. Marie Curie was born in": the attention model's head 2 in block 3 puts 0.42 of its weight on
"Warsaw", and the model says "Warsaw" at 18.4%; the Hyena model says "Paris" at 3.8%.

**Three models of 110M,** episode 19's recipe (12 blocks of 768 channels, windows of 1,024, 3.6B tokens of
FineWeb-Edu, 36,743 steps of 98,304 tokens), one 4090 each: episode 19's attention model, Hyena, and a hybrid with
attention in blocks 3 and 9 (`--attention-layers 3 9`), measured afterwards on one GPU (`hyena_eval.py`,
`hellaswag.py`, `bits.py --web`):

| | attention | Hyena | hybrid |
|---|---|---|---|
| tokens a second, training | 136K | 82K | 88K |
| GB a GPU, training | 14.5 | 17.6 | 17.0 |
| held-out loss after 3.6B tokens | 3.113 | 3.236 | 3.108 |
| held-out loss, long pages | 2.995 | 3.109 | 2.968 |
| loss in the first and last 64 tokens | 3.32, 2.96 | 3.28, 3.11 | 3.24, 2.95 |
| bits a character, held-out web text | 0.956 | 0.989 | 0.949 |
| HellaSwag, 10,042 examples | 29.45% | 28.94% | 29.41% |
| random tokens twice, second half copied | 27.4% | 0% | 37.5% |
| pass keys found, 256 to 1,016 tokens | 68% | 0% | 58.5% |
| ms a token, writing after 960 | 7.4 | 6.2 | 6.4 |
| cache, bytes a token | 73,728 | 36,864 | 43,008 |

Hyena's held-out perplexity is above attention's at all 132 evaluations from step 4,000 on; the hybrid's is below it
at 121, by about 0.8 for most of the run, and the two end level. The HellaSwag scores are within about what chance
moves a score on 10,042 examples. The writing times are from the three timed together on an idle machine, three
times, in both orders, within 0.6 ms; timed beside other jobs, they came out a millisecond apart. The paper's Hyena
matched GPT on The Pile with 153M parameters against 125M; ours has 111.1M against 110.2M.

**Recall,** multi-query associative recall from [Zoology](https://arxiv.org/abs/2312.04927) (Arora et al., 2023):
models of two blocks on windows of 256 tokens, trained for 30,000 steps of 64 examples with 4 pairs first and up to
64 by the end (`recall.py`), at three learning rates. Keys answered right at 4 and 64 pairs, at each model's best rate:

| width | attention | Hyena | hybrid |
|---|---|---|---|
| 64 | 100%, 99.9% | 41.5%, 1.1% | 100%, 99.0% |
| 128 | 53.1%, 6.1% | 50.4%, 2.8% | 100%, 99.7% |
| 256 | 51.6%, 6.5% | 53.4%, 4.5% | 100%, 100% |

Attention at 128 and 256 channels stalls at all three rates (3e-4, 1e-3, 3e-3), as Hyena does at all widths: a
stalled model answers between 0.7 and 4.2 keys an example, however many pairs there are. The hybrid's short
convolution hands each token the one before it, so a value carries its key, and its attention block has the matching
left to do. In Zoology, attention solves MQAR from 64 channels up, with four learning rates swept up to 1e-2, so the
stall at the wider two is more likely our training than attention; gated convolutions need at least as many channels
as the window has tokens.

## Running it

```bash
python -m pytest -q tests/test_hyena.py
python hyena_bench.py --lengths 256 1024 4096 16384 65536           # a layer of each, forward and backward
torchrun --standalone --nproc_per_node 1 pretrain_ddp.py --steps 36743 --web --compile --block 1024 \
  --batch 24 --accum 4 --hyena --name web-hyena                      # episode 19's recipe, with Hyena
torchrun --standalone --nproc_per_node 1 pretrain_ddp.py --steps 36743 --web --compile --block 1024 \
  --batch 24 --accum 4 --hyena --attention-layers 3 9 --name web-hybrid   # and the hybrid
CUDA_VISIBLE_DEVICES=1 ./hyena_runs.sh                               # the six small models, half an hour each
python hyena_eval.py checkpoints/small-*-4096/model.pt               # loss by position, copying, pass keys, writing
python hyena_eval.py checkpoints/pretrain-web/model.pt checkpoints/web-hyena/model.pt checkpoints/web-hybrid/model.pt
python hellaswag.py checkpoints/web-hyena/model.pt                   # and bits.py MODEL.pt ... --web
python flow.py ATTENTION.pt HYENA.pt "Marie Curie was born in Warsaw. Marie Curie was born in" --layer 3 --head 2
python recall.py --steps 30000 --lrs 1e-3                            # multi-query associative recall
```
