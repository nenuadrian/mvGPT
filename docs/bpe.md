# A BPE tokenizer, in C++

mvGPT's first tokenizer splits text into words and runs of punctuation, and keeps the 10,000 most
frequent: a word outside them becomes `[UNK]`, and spaces and new lines are dropped, then put back by
rules when the text is decoded. Byte-pair encoding (BPE), the tokenizer of GPT-2 and most models
since, starts from the 256 possible bytes instead, so any text can be written, and learns merges:
the pair of tokens that appears most often next to each other becomes a new token, again and again.

```
o n → on          on c → onc          onc e → once          ␣ once → ␣once
```

## In the code

`bpe/bpe.cpp` is the tokenizer, in C++, built into a Python module with pybind11:

```bash
python setup.py build_ext --inplace    # builds bpe.*.so
pytest tests                           # the C++ against bpe_reference.py, in plain Python
python bpe_bench.py                    # how much faster the C++ is
```

- `pretokenize` splits text into chunks that merges never cross: a word or a run of punctuation,
  each with the space in front of it, or a run of whitespace. So " upon" can become one token, but
  "time," can't.
- `train` counts each distinct chunk once, with how often it appears, and keeps the counts of the
  pairs in a heap. A merge updates the counts of the chunks that hold the pair, instead of counting
  the whole text again.
- `Encoder.encode` applies the merges to each chunk in the order they were learned, and remembers
  the chunks it has seen.

`bpe_reference.py` is the same algorithm in about fifty lines of Python, and the tests check that
both give the same chunks, the same merges and the same tokens, and that decoding gives back the
text, accents and emoji included. `bpe_bench.py` times both on the same stories, on an Apple M4 Pro:

| | Python | C++ | Faster |
| --- | --- | --- | --- |
| Training 500 merges, 1 MB | 2.29 s | 0.020 s | 116× |
| Encoding 1 MB | 0.47 s | 0.011 s | 42× |

The C++ learns the tokenizer's 7,934 merges from 158 MB of stories and instructions in about one
and a half seconds.

`BPETokenizer` in `tokenizer.py` wraps the module for the model: the 256 bytes, the merges, then
padding and end of text. Checkpoints keep the merges, so a model loads with its tokenizer.

```bash
python data.py --bpe              # learns 7,934 merges: a vocabulary of 8,192
python pretrain.py 10 --rope --bpe    # to checkpoints/pretrain-rope-bpe
python bits.py checkpoints/pretrain/model.pt checkpoints/pretrain-rope/model.pt checkpoints/pretrain-rope-bpe/model.pt
```

## Comparing tokenizers

A model's loss is per token, and two tokenizers cut the same text into different numbers of tokens,
so their losses can't be compared. `bits.py` measures bits per character instead: the loss summed
over the first two million characters of the held-out stories, in bits, divided by the characters.

Ten minutes of pretraining each, on an Apple M4 Pro:

| | Words, learned positions | Words, RoPE | BPE, RoPE |
| --- | --- | --- | --- |
| Vocabulary | 10,000 | 10,000 | 8,192 |
| Parameters | 14,585,856 | 14,487,552 | 13,793,280 |
| Characters per token | 4.16 | 4.16 | 4.01 |
| Unknown tokens | 0.09% | 0.09% | none |
| Stories given back exactly by decoding | 0.6% | 0.6% | 100% |
| Held-out loss, per token | 2.25 | 2.18 | 2.12 |
| Bits per character | 0.782 | 0.744 | 0.763 |

The word-level model has an easier job here: it doesn't predict spaces and new lines, which its
rules put back, or the spelling of a word it doesn't know, which is a single `[UNK]`. So 99.4% of the
held-out stories don't come back as they were, mostly for their new lines. The BPE model predicts each byte,
and with a smaller vocabulary and fewer parameters, it ends about 3% above the word-level model with
the same positions, while it can encode any text.
