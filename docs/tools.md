# Tool use: a calculator

The model [pretrained on the web](web.md) writes fluent text but can't do arithmetic: with three
worked examples in front of it, it gets none of fifty two-digit additions right. So we teach it to
call a calculator, as [Toolformer](https://arxiv.org/abs/2302.04761) did, and leave the arithmetic
to the calculator.

## A call, as plain text

```
Question: A box holds 24 pencils. How many pencils are in 37 boxes?
Answer: <<calc 37 * 24>> = 888. There are 888 pencils in 37 boxes.
```

The model writes `<<calc 37 * 24>>`. `generate_with_tools` in `tools.py` watches each new token, and
when the text ends with `>>`, gives the arithmetic to the calculator. It appends ` = 888` to the
text and to the tokens the model reads next, then lets the model carry on. The markers are plain
text, so the vocabulary, and the export to Hugging Face, don't change. Bigger models use tokens of
their own for this (Llama 3.1's `<|python_tag|>`).

The calculator reads the call with Python's `ast` and takes numbers, `+ - * /`, brackets and minus
signs. A power, a function, a name, an import, a division by zero or a syntax error gives back
`error`, so a call can't run other code. Whole results are written whole, others to 4 decimal places.

```bash
python -m pytest tests/test_tools.py
python tools.py checkpoints/tools/model.pt "What is 48213 * 907?"
python tools.py checkpoints/tools/model.pt "What is 48213 * 907?" --no-tool   # the model writes the result itself
```

## Fifty thousand questions

`tool_data.py` generates them with a seed, into `data/tools/`:

| Kind | Training examples | What |
| --- | --- | --- |
| Arithmetic | 25,000 | ten wordings for each of `+ - * /`, numbers of 1 to 6 digits |
| Word problems | 12,500 | twenty kinds, one or two operations |
| Numbers, no call | 5,000 | "Which is bigger, 347 or 29?" |
| Web pages | 7,500 | from the last FineWeb-Edu file, which pretraining didn't read |

The held-out sets, 200 examples each, are the training questions with new numbers (*seen*),
arithmetic in two new wordings per operation (*phrasing*), six new kinds of word problem (*word*),
questions with numbers that need no call (*numbers*), and the start of web pages, cut mid-sentence
(*web*).

## Fine-tuning, with a mask

`tool_sft.py` fine-tunes `pretrain-web` the way `sft.py` does, with a mask per token. The loss
counts the call and the sentence after the result. It skips the question, which the model is
given, and the calculator's result: a model trained to predict that would be learning arithmetic,
the job we hand to the calculator. Each segment of an example is encoded on its own, which gives
the tokens of the whole text, as each starts at a space or punctuation.

Ten minutes on an Apple M-series laptop, batches of 32 at a learning rate of 1e-4: 500 steps, 16,000
examples, a third of the training set.

## Results

`tool_eval.py` answers each held-out question with the calculator, then with it off, greedily. A
call is right if the calculator gets the answer from it; an answer is right if the sentence after
the result has it.

| Held-out set | Pretrained, 3 examples in front | Fine-tuned | Fine-tuned, calculator off |
| --- | --- | --- | --- |
| Seen wordings, new numbers | 3.5% | 99.5% | 2.0% |
| New wordings | 1.5% | 54.5% | 3.5% |
| New word problems | 1.0% | 40.0% | 1.5% |

On questions that need no call, the fine-tuned model calls 5.5% of the time (the pretrained one
17%), and gets 72% right. On web pages it carries on writing, and calls in 1 of 200. With the
calculator, the answers to the training's questions stay right from 2 digits to 6; without it, the
model gets a few small ones and none with 5 digits. Its one miss there is a 9-digit number
copied with a digit missing.

The wrong calls on new wordings pick the wrong operation, or copy words into the call:

```
If 2479 is shared equally among 37, how much does each get?    calc 2479 - 37
How much are 4656 lots of 58916?                                calc 4656 lots of 58916
64442 marbles are put into 9206 equal bags. How many ...       calc 64442 marbles - 9206
```

It copies numbers well and knows when to call, but it picks the operation from the wording it
learned. With the calculator off, it writes the call again and answers with a number from the
question.

## What's real, and what's scaled down

The loop is how assistants call tools: the model writes a call, the program runs it and puts the
result in the context, and the tool's output is left out of the loss. The scale is not: one tool
and about fifty wordings, where assistants choose between many tools described in the prompt,
with JSON arguments and tokens of their own, fine-tuned on a far wider range of requests.
