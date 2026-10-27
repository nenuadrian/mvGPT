# Constrained decoding

[The agent's graph role](graph-agent.md) copies names from its task into a Cypher query, and some
it copies wrong: asked how much the Okapi weighs, it writes `{name: 'Oakapi'}`, and sg finds
nothing. Here nothing is trained. Before each token of a name in a query, the tokens that would take
the name off a list get a logit of minus infinity, so the model picks the likeliest of the rest.
That's constrained decoding, which is how production models keep a response to a JSON schema or a
grammar.

```bash
python -m pytest tests/test_constrain.py
python constrain.py checkpoints/graph-agent/model.pt "How much does the Okapi weigh?"   # names copied from the task
python agent.py checkpoints/graph-agent/model.pt "Are the Okapi and the Giraffe in the same family?" --names task
```

## A mask on the logits

At each step, the model gives each of its 32,768 tokens a logit, and softmax turns them into
probabilities. A mask adds 0 to the logit of each token that's allowed, and minus infinity to the
rest:

$$p_i = \frac{e^{z_i + m_i}}{\sum_j e^{z_j + m_j}}, \qquad m_i = 0 \text{ if allowed, } -\infty \text{ if not}$$

$e^{-\infty}$ is 0, so the tokens that aren't allowed get no probability, and the allowed ones share
the model's probability out in the proportions it gave them. The weights don't change. Greedy
decoding then takes the likeliest allowed token (`constrain.pick`).

## Tokens aren't letters

The list is of names, and the model writes tokens of any length: `O`, `ak`, or `i'})`, the end of
a name with the quote, brace and bracket after it. `constrain.py` keeps two tries of bytes: the
vocabulary's, with each token's id where its bytes end, and the names'. From the node the name has
got to, `allowed` walks both together, a byte at a time, and a token is allowed while its bytes stay
on a name. At a name's end, the closing quote is allowed, and anything after it in the same token.
Inside single quotes, a name's apostrophe isn't allowed, as it would close the name early
(`Turner's Eremomela` goes in double quotes). The text is kept as the tokens' bytes, as a token can
end inside a character: é is two bytes.

The constraint is on inside a query's `{name: '` (or `{name: "`) and off once the quote is closed.
`Agent(..., constrain=)` takes a function from a context's role and task to a constraint, so a graph
subagent gets one for its own task, and the agent and the other roles write freely.

## Two lists

**The graph's names** (`--names graph --facts`): the 21,834 names of smollerGraph's nodes, from its
facts file. They keep a name real, not right. The model's likeliest token after `'O` is `ak`, at 25%,
and the graph has an Oak Titmouse:

```
$ python constrain.py checkpoints/graph-agent/model.pt "How much does the Okapi weigh?" --names graph --facts ../smollerGraph/data/wikidata.txt
name so far           most likely        allowed  their share   picked
'                     'O'         7.7%       621        18.1%   'O'
'O                    'ak'       25.0%       139        33.3%   'ak'
'Oak                  'ap'       66.1%         4         0.0%   ' T'
'Oak T                'ap'        8.6%         2         0.0%   'it'
'Oak Tit              "'"         3.3%         2         0.0%   'm'
'Oak Titm             "'"        75.3%         4         0.2%   'o'
'Oak Titmo            "'"       100.0%         3         0.0%   'us'
'Oak Titmous          "'"       100.0%         1         0.0%   'e'
'Oak Titmouse         "'"        90.9%         5        91.0%   "'"
```

After `'Oak`, the model's likeliest token is `ap`, for "Oakapi", and the allowed tokens have almost
none of its probability between them; at each step after that, it would close the quote, and the list
makes it carry on.

So the Okapi's weight is asked for the Oak Titmouse, which the graph has no weight for, and the
agent says "The graph doesn't say how much the Okapi weighs". An animal the graph hasn't got has to
become one it has: "Which family does the Marine iguana belong to?" asks for the Marine otter, and
gets Mustelidae.

**The task's words** (`--names task`): a name has to be copied from the task, as a run of whole
words (`constrain.runs`). After `'O`, two tokens are allowed, with about 0.1% of the model's
probability between them, and the likeliest of them is `k`. After `Ok`, the model writes `ap`, `i`
and the quote, each at about 100%, and the query asks for the Okapi. A Komodo dragon can still be
copied, for sg to find nothing. With the graph's names, "Is the Komodo dragon a mammal or a bird?"
asks for the Komodo rat instead, and the answer is that the Komodo dragon is a mammal.

## Results

`constrain_eval.py` asks the graph role [episode 24](graph.md)'s held-out questions three ways, and the
agent [episode 25](graph-agent.md)'s held-out tasks two ways, greedily, through `agent.py` and `sg serve`.
*Off the graph* is how often a query named something the graph hasn't got: a name copied wrong, or
an animal it doesn't hold, as in the *missing* set.

```bash
python constrain_eval.py checkpoints/graph-agent/model.pt ../smollerGraph/data/wikidata.txt   # with sg serve running
```

| The graph role | Free | The graph's names | The task's words | Off the graph |
| --- | --- | --- | --- | --- |
| Seen wordings | 88.0% | 93.0% | 96.0% | 9% / 1% / 0% |
| New wordings | 76.5% | 80.5% | 82.0% | 14% / 0% / 2% |
| Animals the graph hasn't got | 93.0% | 13.0% | 96.5% | 100% / 2% / 100% |

| The agent | Free | The task's words | Facts right |
| --- | --- | --- | --- |
| Seen wordings | 88.0% | 95.5% | 88% / 94% |
| New wordings | 34.0% | 36.0% | 68% / 72% |
| A weight or a species missing | 89.0% | 95.0% | 89% / 94% |

The graph's names help where a name was copied wrong and lose the animals the graph hasn't got: the
Marine iguana becomes the Marine otter, "in the family Mustelidae". The task's words keep both. In the
new wordings, most misses left name an animal the graph has: they go wrong elsewhere, and for the
agent, mostly in reading a new wording as a task it knows, as in [episode 25](graph-agent.md#results).
A question took about half a second in each mode (0.41 to 0.51 s for the graph role, about a second
for the agent), the masks too cheap to see next to the model's passes.

Copying slips aren't all in names. "How much heavier is the Okapi than the Baikal Seal?" gets both
weights right now, then the agent copies 90000 as 9000 into `<<ask math: What is 250000 - 9000?>>`,
and the math role writes "The answer is 24tern" for the calculator's 241000.

## What's real, and what's scaled down

A mask on the logits at each step is how structured outputs keep a model to a JSON schema, and the
tokens are checked against it in bytes, as a token rarely lines up with what's allowed. What's
scaled down is what's kept to a list: the names in one kind of query, where a grammar would cover the
whole query. Libraries such as [Outlines](https://arxiv.org/abs/2307.09702) turn a regular expression
or a grammar into a state machine and build an index over the vocabulary once, so each step is a
lookup; here the tries are walked in Python at each step, about a millisecond at the opening quote.
