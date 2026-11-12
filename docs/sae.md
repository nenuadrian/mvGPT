# Features, with a sparse autoencoder

A sparse autoencoder (SAE) writes each token's residual stream at one point in the model as a sum of a few
directions out of many, the features, trained to rebuild the stream. If it works, a feature tends to mean one thing,
where a single channel of the stream rarely does. `sae.py` trains one on episode nineteen's web model (12 blocks of
768, a BPE vocabulary of 32,768, pretrained on FineWeb-Edu), on the stream after block 6.

```bash
python sae.py norms checkpoints/pretrain-web/model.pt          # the stream's length, position by position
python sae.py train checkpoints/pretrain-web/model.pt          # 15 minutes; checkpoints/sae/sae.pt
python sae.py evaluate checkpoints/sae/sae.pt                  # the model's loss through the autoencoder
python sae.py features checkpoints/sae/sae.pt "The Eiffel Tower is in Paris."
python sae.py show checkpoints/sae/sae.pt 123 456              # where features fire, and what they raise
python sae.py steer checkpoints/sae/sae.pt 123 "The history of"
python -m pytest tests/test_sae.py
```

## The stream

`stream(model, idx, layer)` runs the embeddings and the first `layer` blocks and stops, which halves the work at block
6 of 12. `sae.py norms` averages the stream's length at each position over held-out windows: the first position's is
many times the others', the attention sink (Xiao et al., 2023; "massive activations", Sun et al., 2024). Left in, those
tokens would take over the scale and the loss, so training, evaluation and the top contexts leave each window's first
token out.

On the web model, after block 6, averaged over 16 held-out windows: position 0 is about 559 long, position 1 about 83,
position 2 about 61, and from position 6 on the median is about 37.5, so the first token is about 15 times a typical
one. Positions 1 and 2 are longer too, and are kept.

## The autoencoder

`SparseAutoencoder(d_model, n_features, k)`: an encoder from the stream minus a learned bias to a score per feature,
the `k` largest kept through a ReLU and the rest set to zero (TopK, Gao et al., OpenAI, 2024), and a decoder that adds
up the kept features' directions and the bias. The directions are kept at a length of 1 after each step, so that how
strongly a feature is there is its activation alone; the encoder starts as the decoder's transpose, so that a feature
first fires along its own direction. The stream is scaled by a fixed `scale`, from a first batch, so that its tokens
are about `sqrt(d_model)` long. With an expansion of 16 and `k=32`, it has 12,288 features, 32 per token.

`sae.py train` takes the stream of random training windows as it goes, 4,096 tokens a step, and minimises the error
as a share of the stream's variance with Adam (2e-4). A feature that hasn't fired in 2 million tokens counts as dead.

Fifteen minutes on a Mac mini with an M4 Pro (shared with another program): 3,709 steps, 15.2M tokens, about 16,900
tokens a second. The unexplained share falls from about 0.77 at step 100 to about 0.18, so it rebuilds about 82% of the
stream's variance, and no feature went dead.

## What it costs the model

`Splice(model, layer, change)` is a forward hook, while it's open, that passes the block's output through `change`.
`sae.py evaluate` measures the next-token loss on held-out windows with the model as it is, with the stream replaced by
the autoencoder's reconstruction, and with the stream's average token, and the share of the gap between the last two
the reconstruction recovers.

On 65,536 held-out tokens: 3.080 as it is, 3.295 through the autoencoder, 8.417 with the average token, so the
reconstruction recovers about 96% of the gap.

## Reading and steering

`sae.py features` prints each token of a text with its strongest features; `sae.py show` prints, for feature ids, how
often each fires on held-out text, its strongest contexts, and the tokens its direction raises read straight through
the head. `sae.py steer` adds a feature's direction, times a strength, to the stream while the model writes, and prints
the text with and without it from one seed.

Seven features fire on over half of all tokens (947, 1195, 4690, 5778, 9370, 9547, 10048); `features` leaves out any
feature that fires on over half of 50,000 held-out tokens. Some of the others, from `show`:

| Feature | Fires on | Its strongest contexts | It raises |
|---|---|---|---|
| 1621 | 0.30% | Paris, Atlanta | city, Metro, Metropolitan |
| 10652 | 0.01% | capital (the city) | |
| 1767 | 0.04% | carbon | dioxide, monoxide |
| 5386 | 0.01% | moon, Jupiter's moon Io | |
| 7516 | 0.34% | years: 1932, 1912, 1896 | |
| 6198 | 0.25% | full stops, at activations near 70 | |

Steering with the moon feature (5386) at a strength of 12 turns "My favourite thing to do at night is" towards a huge
moon; at 30, the text breaks into "the moon and the moon". The city and carbon features, at strengths that kept the
text readable, barely pulled it towards cities or carbon.
