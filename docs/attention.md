# Inside the model

The [agent's graph role](graph-agent.md) copies a name from its task into a Cypher query, and
[episode 26](constrained-decoding.md) found where it copies wrong. Here we look inside the model
while it copies: what each attention head looks at, and what each layer would write next. Then we
turn heads off, to test what we saw.

```bash
python -m pytest tests/test_interpret.py
python interpret.py checkpoints/graph-agent/model.pt lens "How much does the King Penguin weigh?" " <<cypher MATCH (s:Species {name: '"
python viewer.py checkpoints/graph-agent/model.pt "How much does the King Penguin weigh?" " <<cypher MATCH (s:Species {name: 'King Penguin'}) RETURN s.weight_g>>" --heads results/heads-graph-agent.json
python interpret.py checkpoints/graph-agent/model.pt heads   # with sg serve running
```

## Attention, written out

A head's weights are $A = \text{softmax}(QK^\top / \sqrt{64} + M)$, a row per token, each row's
weights zero or more and adding up to one over the tokens before it and its own place.
`F.scaled_dot_product_attention` doesn't hand them back, so with `CausalSelfAttention.keep` on,
attention is written out as [episode 13](flash-attention.md)'s `attention_by_hand` does, and the
weights are kept. `interpret.attention` returns them for each layer and head: layers, heads, tokens,
tokens. `CausalSelfAttention.off` lists heads whose output is zeroed, and `interpret.HeadsOff` turns
heads off while it's open.

## The logit lens

The residual stream is the sum each block adds to, and the final norm and the head turn the last of it
into logits. The logit lens ([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens))
puts them on the stream after the embeddings and after each block, so each layer's guess at the next
token shows. `interpret.py lens` prints the three likeliest tokens a layer, at the end of an answer so
far.

| After `{name: '` | Layer 9 | Layer 10 | Layer 11 | Layer 12 |
| --- | --- | --- | --- | --- |
| The King Penguin | `F` 1.7% | `T` 5.7% | `T` 4.9% | `King` 96.0% |
| The Okapi | `B` 3.0% | `K` 20.7% | `K` 14.7% | `O` 7.7% |
| The Okapi, after `'O` | `ak` 2.2% | `ak` 11.1% | `ak` 4.9% | `ak` 25.0% |

The King Penguin's name shows up in the last layer and in no layer before it. The model is unsure of
the Okapi from the start, and after the `O` it writes, the `k` the name needs comes out as `ak`.

## A page to look with

`viewer.py` writes one HTML file, with the numbers in it, for a prompt and an answer: the tokens,
shaded by how much the token picked takes from each, for a head or a whole layer averaged, with the
weights of 0.1 or more under them; the 144 heads' patterns in a grid; and the logit lens at the token
picked. Each weight is a byte, so a prompt's heads fit in under a megabyte.

On the King Penguin's query:

- **A previous-token head:** layer 10, head 1 puts 0.95 of its attention, from `Penguin` in the query, on
  the token before it, `King`.
- **A resting head:** layer 9, head 5 puts most of its attention on the first token, `R`, as many heads do.
- **Heads that look at the name:** from the name's opening quote, layer 12, head 1 puts 0.92 on `King` in the
  task, and from `King`, 0.97 on `Penguin`, the next word of the name. Head 4 puts 0.96 on `King` too, though
  over the questions below it scores 58%, to head 1's 92%. Averaged over layer 12's heads, half the attention
  rests on `R` and a third on the line break after the role's line, and the name gets 0.16, nearly all of it
  from those two heads.

On the Okapi's query (`'Oakapi'`), that head puts 0.94 on ` Ok` in the task from the opening quote,
the right place, and head 4 puts 0.95 there. As with `King`, the name in the task has its word's space at
the front, and inside the quotes the model needs the token without it: `King` gets 96% there, and `Ok`
gets 0.009%, so the model writes `O`. From the `O`, the head moves on to `ap`, with 0.72, as if the `k`
were written, and the lens says `ak`. From the quote, the head looks at the right place; it's the tokens
that go wrong, and the head follows them.

## Heads turned off

`interpret.py heads` scores each head by the share of its attention that goes to the name in the task,
from the tokens that write the name into the query (the opening quote and each name token but the
last), over episode 24's 200 held-out questions in the training's wordings. The top: layer 12, head 1
(92%), layer 11, head 5 (86%), layer 12, head 8 (81%), layer 12, head 7 (78%). Then the graph role asks
the 200 questions with the top heads turned off, and with as many random heads, three draws each:

| Heads off | The top ones: right | Names right | Random ones: right | Names right |
| --- | --- | --- | --- | --- |
| 0 | 88.0% | 91% | | |
| 1 | 85.5% | 87% | 87.8% | 91% |
| 2 | 84.0% | 86% | 87.8% | 91% |
| 4 | 72.0% | 85% | 87.7% | 91% |
| 8 | 0.0% | 2% | 78.8% | 89% |

*Names right* is the share of the queries that name something that name the right thing. With the top
four off, the role loses 16 points, and the names it writes lose 6: other heads carry the name too, and
these heads do other work. With the top eight off, it still names something in 134 of its 200 answers,
and 3 of those names are right, so between them the eight do carry the name; some answers write no query
at all (`<<calc the -i +z>>`). The random heads are drawn from the whole model, and of their 24 heads for
eight off, 4 are in the last two layers, where the top eight have 7: so this can't tell heads that look at
the name from heads in the last layers. A head's pattern is a guess at what it does, and turning it off
tests the guess.

## What's real, and what's scaled down

Attention patterns, the logit lens and turning heads off are how a lot of interpretability work on
transformers starts, and previous-token heads and heads that look back at what's being copied are
kinds found in much larger models. What's scaled down: one task, heads turned off a group at a time,
and no path traced from one layer to the next, as work on circuits and features in the residual stream
does.
