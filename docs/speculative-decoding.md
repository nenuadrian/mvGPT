# Speculative decoding

The [agent](graph-agent.md) writes a token per forward pass. At batch 1 on a GPU, a pass costs about
the same for a few tokens as for one, as with a model this small, most of its time goes into launching its
operations. Speculative
decoding ([Leviathan, Kalman and Matias, 2023](https://arxiv.org/abs/2211.17192); [Chen et al.,
2023](https://arxiv.org/abs/2302.01318)) uses that: a cheap draft guesses the next few tokens, and the
model checks them all in the pass that writes its own. With greedy decoding, the text is the one the
model writes a token at a time, in fewer passes.

```bash
python -m pytest tests/test_speculative.py
python speculative.py checkpoints/graph-agent/model.pt "How much does the King Penguin weigh?" --draft lookup
python agent.py checkpoints/graph-agent/model.pt "How much heavier is the Okapi than the Lion?" --names task --draft checkpoints/draft/model.pt
```

## What a pass costs

`speculative_bench.py` times a pass of 1 to 16 new tokens after 100 in the KV cache, the median of
40, on the Mac's GPU and its CPU (an Apple M4 Pro, with nothing else running):

| Tokens in the pass | 1 | 2 | 5 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| GPU (MPS), ms | 6.8 | 6.7 | 9.6 | 8.9 | 11.0 |
| CPU, ms | 6.4 | 15.9 | 15.0 | 14.8 | 16.4 |

On the GPU, with a model this small, most of a pass at batch 1 goes into launching its operations,
as [episode 4](kv-cache.md) found, so a few more tokens cost little. On the CPU, one token is about as quick, then two or more take about two
and a half times as long, and about that up to sixteen.

```bash
python speculative_bench.py checkpoints/graph-agent/model.pt
```

## Guesses, checked in one pass

`speculative.passes` decodes greedily a pass at a time. Each pass takes the model's own next token,
from the logits of the pass before, and the draft's guesses at the tokens after it, through the model
together. The argmax at each position is the token the model writes after it, so a guess is kept
while it's that token: the first if it's what the model writes after its own, the second if it's what
it writes after the first, and so on. The guesses it didn't keep come off the KV cache
(`KVCache.trim`), and the logits after the last one kept give the next pass its token.

Two drafts:

- **Lookup** (prompt lookup decoding): the tokens that came after the last time the text's last
  three, two or one tokens were written. It costs nothing, and catches what an answer copies from its
  task and from the tools' results. It can't guess what isn't in the text, such as the query's template,
  and a name in quotes starts without its space, so it's a different token from the task's.
- **A draft model** (`draft_sft.py`): a GPT of 4 layers of 384 channels (19.7M parameters, against the
  agent's 110M), with the agent's tokenizer, trained from scratch for 30 minutes on episode 25's
  training set, the prompts and the tools' results masked out of the loss. It keeps a KV cache of its
  own, and drops what the model turned down.

```bash
python draft_sft.py 30   # needs data/graph-agent/train.jsonl, from graph_agent_data.py
```

In 30 minutes on the Mac: 7,954 steps, 5.54 passes through the 44,996 examples, a held-out loss
of 0.30, and 95.8% agreement with the agent's held-out tokens (81.3% at step 500).

The graph role's query for the King Penguin's weight, 32 tokens, a pass a line
(`speculative.py ... --draft`):

```
lookup: 31 passes, 1 of 16 guesses kept        the draft model: 7 passes, 26 of 28 kept
  [ <]                                            [ <]<cypher
  [<]                                             [ MAT]CH (s:
  [c]                                             [Spec]ies {name
  [yp]her                                         [:] 'King Penguin'
  [ MAT]                                          [}]) RETUR
  ...                                             [N] s.weight_
                                                  [g]>>
```

Lookup finds `cypher` in the prompt's line of tools and nothing else: the query's template isn't in
the prompt, and `King` in quotes is a different token from the task's ` King`. The draft model knows
the template.

## In the agent

`Agent(..., draft=)` checks the guesses through episode 26's constraint, so a guess that would take a
name off its list is turned down, and writes the kept ones a token at a time: a kept guess that closes
a call runs it, and the guesses after it are dropped. With no draft, the agent makes the passes it
did before.

`speculative_eval.py` runs episode 25's 600 held-out tasks through the agent with names copied from
the task, with no draft, with lookup and with the draft model, k = 4:

```bash
python speculative_eval.py checkpoints/graph-agent/model.pt checkpoints/draft/model.pt   # with sg serve running
```

| Draft | Right | Traces as greedy's | Passes a task | Guesses kept | Seconds a task | Faster |
| --- | --- | --- | --- | --- | --- | --- |
| None | 75.5% | 100% | 142.9 | | 1.00 | 1.00× |
| Lookup | 75.5% | 100% | 100.6 | 30.4% | 0.79 | 1.27× |
| The draft model | 75.5% | 100% | 40.3 | 78.7% | 0.69 | 1.46× |

Each trace, the agent's and its subagents' calls, results, answers and lengths, is greedy decoding's.
The draft model takes 3.5 times fewer passes and 1.46 times less time: a pass with four guesses takes
longer than a pass of one token, and the four guesses are four passes of the small model, whose time
at batch 1 goes with its four layers more than with its size.

## What's real, and what's scaled down

Checking a draft's guesses in one pass is how serving systems such as vLLM speed up a model at small
batches, with prompt lookup and small draft models among their drafts. Here decoding is greedy; with
sampling, the papers keep a guess with a probability from the two models' probabilities, and at the
first one turned down, draw the model's token from what's left of its distribution, so the text keeps
the model's distribution. And the draft guesses one line of tokens, where methods such as
Medusa and EAGLE guess a tree of them, with small heads on the model, and check the tree in one pass.
