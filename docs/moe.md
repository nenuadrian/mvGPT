# A mixture of experts

Each block's MLP can be a mixture of experts: several smaller MLPs, the experts, and a router, a linear layer that
scores the experts for each token. A token goes through the experts with its top scores, and their outputs are added
up, weighted by the scores. The idea goes back to Shazeer et al.'s sparsely-gated mixture of experts (2017); Mixtral
8x7B made 8 experts with 2 per token a common setup, and DeepSeek-V3 and Qwen3's largest models are mixtures too.

```bash
python -m pytest tests/test_moe.py
python pretrain.py --modern --steps 2000                       # the dense model, for comparison
python pretrain.py --modern --moe 8 --steps 2000               # 8 experts, 2 per token, the balancing loss
python pretrain.py --modern --moe 8 --balance none --steps 2000
python pretrain.py --modern --moe 8 --balance bias --steps 2000
python experts.py checkpoints/pretrain-modern-moe8/model.pt    # which words each expert gets
```

## Sizes

`GPTConfig(n_expert=8)` replaces each block's `MLP` with `MoE`. `expert_top_k` is how many experts a token goes to
(2), and `expert_hidden` an expert's width, by default the MLP's divided by `expert_top_k`: with SwiGLU's 1,024
channels, 512. So a token goes through as many weights as with one MLP, about 1.18M a block, while a block holds four
times as many, 4.72M. Our model goes from 14.6M parameters to 35.8M, with 14.6M used per token. `n_expert=0`, the
default, keeps the MLP, and older checkpoints load as they were. The experts are SwiGLU MLPs, so a mixture needs the
modern block (`--modern`).

## Routing

The router's scores go through a softmax, `torch.topk` picks each token's two highest, and the two weights are scaled
to add up to 1, as in Mixtral. The (token, expert) pairs are sorted by expert, so that each expert's tokens are one
block of rows.

The first version ran each expert as an `MLP` in a `ModuleList`, one matrix multiply per expert. The experts' weights
are now stacked, `gate` and `up` as experts x 384 x 512 and `down` as experts x 512 x 384, the rows go into one
block per expert, padded with zeros to the busiest expert's count, and `torch.bmm` runs all the experts at once.
`MoE.expert(e, x)` runs one expert on its own, for the tests. `tests/test_moe.py` checks each token's output against
the weighted sum of its two experts written out token by token.

## Balancing

A router left alone tends to send most tokens to a few experts, which then get the most training and are picked
even more. `pretrain.py --balance` picks how that's prevented:

- `loss` (the default): Switch Transformer's balancing loss, `n_expert` times the sum over experts of the share of
  the pairs each got times its average score, which is 1 when the load is even. It's averaged over the blocks and
  added to the next-token loss at a weight of 0.01. The loss printed and logged is the next-token loss alone.
- `bias`: DeepSeek-V3's balancing without a loss. Each expert has a bias, a buffer, added to the scores to pick the
  experts and nowhere else, so the weights still come from the scores. After each optimiser step, `update_bias`
  lowers the bias of each expert that got more than its share of the batch by 0.001, and raises the others'.
- `none`: neither.

Each 50 steps, TensorBoard gets each expert's share of the pairs in each block (`experts/layerL/expertE`), and the
printed line gets the smallest and largest share: 12.5% each when 8 experts are even.

## Runs

Four runs of `pretrain.py --modern --steps 2000` on a Mac mini with an M4 Pro: 2,000 steps of 48 x 256 tokens,
24.6M tokens, about 1.24 times TinyStories' 19.85M training tokens, in random windows.

| Run | Parameters | Held-out loss | Minutes | Busiest expert, at the end |
|---|---|---|---|---|
| Dense | 14.6M | 1.904 | 19.9 | |
| 8 experts, balancing loss | 35.8M (14.6M a token) | 1.883 | 29.6 | about 20% |
| 8 experts, bias | 35.8M | 1.890 | (the GPU was shared) | about 13% |
| 8 experts, no balancing | 35.8M | 1.905 | 36.0 | about 41%, with experts at 0% |

- Per step, both balanced mixtures end below the dense model, within a hundredth of each other, which is about as
  much as the held-out estimate moves between batches. The gap was widest at step 500, about 0.07.
- The mixture's training loss ends 0.05 below its held-out loss, three times the dense model's gap: with 2.5 times
  the weights, it starts to memorise a dataset this small.
- Per minute, the dense model wins: after its 19.9 minutes, the mixture with the loss was at step 1,364, at about 1.97.
  On this Mac's GPU, a mixture's step takes about 1.4 times the dense model's when the experts are batched, and more
  as a loop; without balancing, the padding to the busiest expert makes it slower still.
- Without balancing, the busiest expert in a block takes about 41% of the pairs, some experts get almost none after the
  first few hundred steps, and the loss ends at the dense model's.
- The bias starts as lopsided as the rest and gets the experts close to even within a few hundred steps.

## Which words each expert gets

`experts.py` routes the first 100,000 held-out tokens and prints each block's share of the first choices by expert,
the eight words each expert gets most often in one block (the middle one, by default), each with the share of its uses
that went to that expert, and a held-out story with each word in the colour of its first expert.

With the balancing loss, the experts split the words by kind. In block 3, one gets the words that start a clause (and,
they, he, she), one the full stops, exclamation marks and closing quotes, one mostly nouns (mom, dog, bird, along with
big) and one feelings (happy, sad, fun, smiled), most of their uses each. With `--layer 0`, most words go to one
expert nearly each time; with `--layer 5`, the shares are lower, so the words around a word count for more. Without
balancing, the experts specialise too, while some of them get almost no tokens.
