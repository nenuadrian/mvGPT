# The agent asks the graph

The [agent](agent.md) hands work to subagents: math, with the calculator, and files, in a sandbox.
[Last episode](graph.md)'s model learned to write Cypher queries for smollerGraph, but on its own,
one question at a time. Here the query becomes a fourth role of the agent, so it can take facts from
the graph and do something with them: asked how much heavier the King Penguin is than the Great White
Pelican, it asks the graph for each weight, and math for the difference.

```
Role: agent
Tools: ask math, ask files, ask graph, write, list
Task: How much heavier is the King Penguin than the Great White Pelican?
Answer: <<ask graph: How much does the King Penguin weigh?>> = The King Penguin weighs about 12150 grams. <<ask graph: How much does the Great White Pelican weigh?>> = The Great White Pelican weighs about 9520 grams. <<ask math: What is 12150 - 9520?>> = 2630. The King Penguin is about 2630 grams heavier than the Great White Pelican.
```

Each `ask graph` is a subagent in a context of its own, which writes last episode's query:

```
Role: graph
Tools: cypher
Task: How much does the King Penguin weigh?
Answer: <<cypher MATCH (s:Species {name: 'King Penguin'}) RETURN s.weight_g>> = 12150. The King Penguin weighs about 12150 grams.
```

## A fourth role

| Role | Tools |
| --- | --- |
| agent | `ask math`, `ask files`, `ask graph`, `write`, `list` |
| math | `calc` |
| files | `read`, `write`, `list`, `calc` |
| graph | `cypher` |

A `cypher` call goes to `graph.run_query`, so `sg serve` reads it with last episode's limits: it
can't change the graph, a query over 300 characters isn't sent, and an answer shows 5 rows and
"and 12 more". `agent.py --server` and `Agent(..., server=)` say where `sg serve` is listening, or
`SMOLLERGRAPH` does, as for `graph.py`.

The graph has no calculator, and the agent can't write a query: each subagent does one job, and the
agent puts the facts together.

**A graph subagent hands back its whole sentence.** Math and files hand back the first number of
their final sentence, which is all a sum needs. The graph's facts are names as often as numbers,
and a list's "and 3 more" would hand back 3. So the agent gets "The King Penguin weighs about 12150
grams", and copies the number into its `ask math` call.

```bash
python -m pytest tests/test_agent.py tests/test_graph_agent.py
python agent.py checkpoints/graph-agent/model.pt "How much heavier is the King Penguin than the Great White Pelican?"
python agent.py checkpoints/graph-agent/model.pt "How much does the King Penguin weigh?" --role graph
```

## The tasks

`graph_agent_data.py` builds 45,000 training examples into `data/graph-agent/`. Each fact in them
comes from a graph subagent's run, built from `sg serve`'s answer by `graph_data.py`'s maker, and
each sum from the calculator, so the examples can't disagree with the database. Each example keeps
its graph subagents' runs under `subagents`, which the tests replay through `agent.py`; training reads
the agent's own trace, and the graph role learns from its 25% share.

| Kind | Training examples | What |
| --- | --- | --- |
| Episode 21's | 20,250 | its math, files, agent and web examples, with the agent's new line of tools |
| The graph role | 11,246 | last episode's questions (numbers aside), under `Role: graph` |
| Heavier | 3,150 | "How much heavier is the Lion than the Tiger?": two graph calls, then `-` |
| Delegate | 2,700 | "Which family does Thalurania watertonii belong to?", handed to the graph as it's asked |
| Same family | 2,250 | "Are the King Penguin and the African Penguin in the same family?": two calls, no calculator |
| Together | 1,800 | "How much do ... and ... weigh together?": two calls, then `+` |
| Kilograms | 1,800 | "How many kilograms does ... weigh?": one call, then `/ 1000` |
| Saved | 1,800 | "What family is ...? Save the answer in notes.txt.": the graph, then `write` |

The agent asks the graph one way for a weight, "How much does the King Penguin weigh?", and one way
for a family, both from last episode's training wordings, so the graph role stays on wordings it was
trained on. New wordings are the agent's to read.

Of the tasks about weights, 15% name a species with no weight on Wikidata and 5% one the graph
hasn't got. The agent stops at that call, and says so: "The graph doesn't say how much the African
Penguin weighs, so I can't answer that." It doesn't make a number up.

The 45% of episode 21's examples keep the agent's three roles. Its held-out wordings ("Write the
answer to ...", "store") stay out of training, so its held-out sets still measure something. The
data took 20,424 distinct queries to `sg serve`, 1,312 of them finding nothing, in about 20 seconds,
and comes to 2.47 million tokens to learn.

The held-out sets, 200 tasks each, are about species and families that training doesn't ask about,
from a random generator of their own: the training's wordings (*seen*), a held-out wording for each
task, with file names training doesn't use (*phrasing*), and a weight or a species missing
(*unknown*).

## Fine-tuning

`graph_agent_sft.py` fine-tunes last episode's model, which already writes the queries, with
`agent_sft.py`'s loop: batches of about 6,000 tokens of like-length examples, a learning rate of
1e-4, and the prompt and the tools' results masked out of the loss.

```bash
python graph_agent_data.py ../smollerGraph/data/wikidata.txt   # needs data/agent/train.jsonl, from agent_data.py
python graph_agent_sft.py 30   # with sg serve running, for the demo
```

Before it, asked as the agent, last episode's model writes a `<<cypher ...>>` call of its own, which
the agent's role doesn't have: `error: no such tool`. Thirty minutes on an Apple M-series laptop:
1,675 steps, 52,552 examples, 1.17 passes over the training set. The held-out loss falls from 3.0 to
0.015 by step 250, and ends at 0.020.

The demo asks about the Okapi and the Baikal Seal, both held out, and at each evaluation, the graph
role copies "Okapi" wrong: `{name: 'Ookapi'}`, then `'Otapi'` and `'Oakapi'`. sg finds nothing, and
the agent says the graph has no Okapi. See below.

## Results

`graph_agent_eval.py` runs each held-out task through `agent.py` and `sg serve`, greedily, and checks
the final sentence and the files, as `agent_eval.py` does: a number has to be in the final sentence,
and any other answer the sentence has to contain. *Facts right* is whether the graph
subagents handed back what the task needs, so a wrong answer with the facts right is the agent's own
slip.

```bash
python graph_agent_eval.py checkpoints/graph-agent/model.pt   # with sg serve running
```

| The new tasks | Right | Facts right | Calls | Subagents | Tokens |
| --- | --- | --- | --- | --- | --- |
| Seen wordings, species training doesn't name | 88.0% | 88.0% | 2.1 | 1.9 | 126 |
| New wordings | 34.0% | 68.5% | 1.6 | 1.6 | 123 |
| A weight or a species missing | 89.0% | 89.0% | 1.3 | 1.3 | 115 |

By task, in the seen wordings: delegate 96%, saved 94%, kilograms 93%, heavier 84%, together 82%,
same family 80%. In the new ones: delegate 83%, heavier 59%, same family 38%, together 3%, kilograms
0%, saved 0%. A question handed to the graph as it's asked is the graph role's to read; for the
others, the agent maps a new wording onto a task it knows, and does that task:

| Held-out wording | What the agent does |
| --- | --- |
| "How much does the Black Guillemot weigh in kg?" | the weight in grams, no division |
| "What would ... and ... weigh on one scale?" | a subtraction, as for heavier: "about -58 grams together" |
| "... Keep the answer in violet7.txt." | the answer, and no file |

| The graph role, on last episode's questions | Right | Episode 24 |
| --- | --- | --- |
| Seen wordings | 88.0% | 89.0% |
| New wordings | 76.5% | 79.5% |

| Episode 21's tasks | Right | Episode 21 |
| --- | --- | --- |
| Math | 99.0% | 100.0% |
| Files | 97.0% | 100.0% |
| Files, new wordings | 11.0% | 49.0% |
| Agent | 97.0% | 100.0% |
| Agent, new wordings | 1.0% | 23.0% |

In the wordings training repeats, the agent is back near where episode 21 left it; in the ones
episode 21 held out, the files role reads a `numbers.txt` that isn't there, and the agent leaves out
the `write` that "store" asks for. Between episode 21's agent and this one there are two fine-tunes,
last episode's and this one, and at step 0 last episode's model, asked as the agent, wrote a query of
its own. Starting from episode 21's agent instead, with last episode's queries as one more role, is
the run that would tell which of the two cost the held-out wordings.

## The Okapi, and other slips

In the task, the tokenizer writes the Okapi as ` Ok`, `ap`, `i`, with the word's space at the start
of its first token, as [last episode](graph.md#a-space-in-the-name) found. Inside the query's
quotes, the first token has to be `Ok`, without the space, and no name in this training's queries
for the graph role starts with it. The model writes `O`, guesses at the rest (`ak`, `ok`, `t`), and copies `ap`
and `i`. Last episode's model, asked `graph.py ... "How much does the Okapi weigh?"`, writes
`'Ookapi'` too.

The math role slips on long sums, which are rare in its training (episode 21's): asked
"What is 1974 - 10?", it calls `<<calc 1974 - 10>>`, gets 1964, and writes "The answer is64".

```
How much heavier is the Lion than the Tiger?
  ... = The Lion weighs about 157000 grams ... = The Tiger weighs about 150000 grams ... = 7000.
  The Lion is about 7000 grams heavier than the Tiger.
```

Wikidata's Tiger is lighter than its Lion, and the agent's answer is as good as the source.

## What's real, and what's scaled down

Splitting the work is how multi-agent systems are built: an agent hands part of a task to a
subagent with the tools for it, and gets back an answer, not the subagent's context. The facts come
from a real database, through queries the model writes, and the sums from a calculator, so the
numbers in an answer are the source's. What's scaled down: the tasks come from templates, in four
or five wordings each, about weights and families; the agent asks the graph in one fixed wording;
and names and numbers have to be copied token by token, which is where most of the misses are.
