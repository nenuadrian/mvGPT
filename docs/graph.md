# A graph database: smollerGraph

The [agent](agent.md) calls a calculator and reads files, but the facts it knows are the ones
training put in its weights, and changing one means training again. Asked the family, the
conservation status or the weight of mammals and birds that training doesn't ask about, with no
database to look in, it gets none of them right. So the facts go in a database, and the model
learns to ask it: it writes a query, the program running it sends the query to the database, and
the model answers from what comes back.

The database is smollerGraph, a graph database built from scratch in C++ in the sibling series,
[Building smollerGraph](https://smollergraph.nenuadrian.com). By its episode 10 it holds the mammals
and birds on [Wikidata](https://www.wikidata.org), and answers a small subset of
[Cypher](https://opencypher.org), the query language of Neo4j and other graph databases.

## The graph

```bash
git clone https://github.com/nenuadrian/smollerGraph && cd smollerGraph && git checkout ep010/end
cmake -S . -B build -G Ninja && cmake --build build   # CMake 3.28 or newer, Ninja and a C++20 compiler
./build/sg load wikidata.sg data/wikidata.txt
./build/sg serve wikidata.sg 8474
```

`data/wikidata.txt` has 21,834 taxa: 17,614 species, 3,722 genera, 423 families, 73 orders and the
two classes, Mammalia and Aves, each joined to the one above it by a `PARENT_TAXON` relationship.
17,603 species have all four ranks above them, one hop each. Each species has its conservation
status, as the IUCN lists it, all but 2 have a scientific name, and 3,817 have a weight in grams.

mvGPT looks for `sg serve` at http://localhost:8474, or where `SMOLLERGRAPH` points.

## A call, as a Cypher query

```
Question: What family is the Red fox in?
Answer: <<cypher MATCH (s:Species {name: 'Red fox'})-[:PARENT_TAXON*2]->(f:Family) RETURN f.name>> = Canidae. The Red fox is in the family Canidae.
```

It's the [calculator](tools.md)'s format, with another tool. The model writes `<<cypher ...>>`;
`generate_with_graph` in `graph.py` watches each new token, and when the text ends with `>>`, sends
what's between the markers to `sg serve`'s `/api/query` over HTTP. It appends ` = Canidae`, sg's
answer as text, and lets the model carry on with the KV cache. Cypher has `>` in its arrows and its
comparisons, but never two together, so only `>>` closes a call.

sg's answer goes in as its first 5 rows, a row's values joined by ` | `, then "and 12 more", so a
long answer can't fill the context. A property a node hasn't got is `unknown`, no rows is
`nothing`, and a query sg can't parse is `error: ` with sg's reason. Nothing `sg serve` does
changes the graph, so a query the model writes can only read, and one over 300 characters is
refused before it's sent.

Names are matched exactly, as Wikidata writes them ("Red fox", "Crimson-breasted Shrike"), so the
questions use the stored name, and the model copies it into the query. A name with an apostrophe
goes in double quotes: `"Finsch's Francolin"`.

```bash
python -m pytest tests/test_graph.py
python graph.py checkpoints/graph/model.pt "What family is the Red fox in?"
python graph.py checkpoints/graph/model.pt "What family is the Red fox in?" --no-tool   # the model writes sg's answer itself
```

## Thirty thousand questions, most answered by the graph

`graph_data.py` reads `data/wikidata.txt` and generates the questions with a seed, into
`data/graph/`. Each query goes to `sg serve` once, and the example is built from its answer, so the
examples can't disagree with the graph: 22,126 distinct queries, 1,811 of them finding nothing, in
about 13 seconds.

| Kind | Training examples | What |
| --- | --- | --- |
| Genus, family, order, class | 11,700 | `[:PARENT_TAXON*2]` to a `Family`, and so on: "Is the Iiwi a bird or a mammal?" |
| Status | 4,200 | "How threatened is the Lion?" |
| Weight | 3,300 | "How heavy is the Iiwi?", and some about species with no weight: "The graph doesn't say..." |
| Scientific name | 2,700 | "What is the Latin name of the Madagascar Cuckoo-Hawk?" |
| Lists | 3,300 | "Which species in Felidae are endangered?", in families and orders, with a `WHERE` |
| Not in the graph | 2,400 | reptiles, fish and insects ("Komodo dragon"), and names made of two real ones |
| Numbers, no call | 2,400 | [episode 20](tools.md)'s "Which is bigger, 584 or 558?" |

Training asks each kind 7 or 8 ways. About an animal the graph hasn't got, sg finds nothing, and
the model learns to say "The graph has no species called Komodo dragon." rather than make an
answer up.

The held-out sets, 200 examples each, come from a random generator of their own, so a change to
what training asks hardly changes them. They are the training's wordings about the tenth of the
species, families and orders that training doesn't ask about, chosen by a hash of the name (*seen*); a
wording of each kind that training never uses, about those species (*phrasing*); animals the graph
hasn't got that training never names (*missing*); and number questions (*numbers*).

## Fine-tuning the agent

`graph_sft.py` fine-tunes the agent, which already writes `<<...>>` calls, with `tool_sft.py`'s loop
and its mask: the loss counts the query and the sentence after sg's answer, and skips the question
and the answer, which the model is given. Before it, asked about the Red fox, the agent writes
`<<read Red fox>>`, then `<<calc red fox - 5>>` over and over: the shape of a call, with the tools
it knows.

Twenty minutes on an Apple M-series laptop, batches of 32 at a learning rate of 1e-4: 1,024 steps,
32,768 examples, a little more than one pass over the training set, with 1.83 million tokens to
learn. The held-out loss falls from 4.7 to 0.017 by step 250, and ends at 0.018.

## Results

`graph_eval.py` answers each held-out question through `sg serve`, then with the model writing sg's
answer itself, greedily. A query is right if sg's answer to it is the right query's, so another
query that finds the same thing counts; an answer is right if the model writes the example's
sentence after sg's answer.

| Held-out set | Calls | Query right | Answer right | Graph off |
| --- | --- | --- | --- | --- |
| Seen wordings, species not asked about | 100% | 92.0% | 89.0% | 0.0% |
| New wordings | 100% | 82.0% | 79.5% | 0.0% |
| Animals not in the graph | 100% | 100% | 98.0% | 0.0% |

With the graph off, the model still writes the query, but no answer comes after it. In training, sg's
answer always came after a call, and the loss left it out, so the model hasn't learned what goes
there: it mostly repeats the end of its query, `RETURN f.name>> RETURN f.name>> ...`, or writes its
sentence with the fact missing, "The White-tailed Blue Flycatcher weighs about grams." It gets none
right: the answers come from the graph. On the number questions it makes no calls,
and gets 71% right, about what episode 20's model got (72%).

By kind, on the seen and new wordings together: weight 96%, status 96%, order 95%, class 94%,
family 89%, genus 83%, lists 82%, and scientific names 36%. The scientific names are 18 of 22 right
in the training's wordings and none of 28 in the new one, "What do biologists call the Grey-breasted
Mountain Toucan?": the model reads it as a question about the weight, in 24 of the 28. Without that
wording, the new wordings are 92% right. Of the 9 lists it gets wrong, 5 have the right query: the
query finds the right species, and the model copies one of the names back wrong.

```
What do biologists call the Grey-breasted Mountain Toucan?
  ... RETURN s.weight_g>> = unknown. The graph doesn't say how muchipaner's Mountain Toucan weighs.
What least concern species does Melanocharitidae have?
  ... = Pygmy Longbill, Slaty-headed Longbill, Satin berrypecker, ... are Pygmy Longbill, Slaty-headed Longbill, Satin Orangepecker, ...
What family of animals is the St. Michel cuscus part of?
  ... = nothing. The graph has no species called St.Micuscus.
```

## A space in the name

The BPE tokenizer from [episode 11](bpe.md) keeps a word's space at its start. In the question, the
Westland Petrel is ` West`, `land`, ` Pet`, `rel`; inside the query's quotes, its first token has to
be `West`, with no space, a different token. Copying the question's token gives
`{name: ' Westland Petrel'}`, which sg doesn't find.

The model does it on 1.5% of the seen questions, 0.5% of the new wordings and 0.5% of the animals
not in the graph, with families as well as species. Asked which species in Turnicidae are
critically endangered, it queries `{name: ' Turnicidae'}`, sg finds nothing, and the model writes
"None of the species in Turnicidae is critically endangered.", where the graph has two. A query
that finds nothing because it's wrong reads the same as one that finds nothing because there's
nothing. A token copied wrong is
the other slip: the Black degu becomes `'Blackhipu'`.

## What's real, and what's scaled down

The loop is how assistants query databases: the model writes a query in the database's own
language, the program runs it and puts the answer in the context, and the answer is left out of the
loss. The database is real too: smollerGraph keeps the graph on disk in pages, finds a name with a
B+ tree, and plans each query. What's scaled down: the questions come from templates, in 57
wordings; the graph has one relationship type, and the queries a fixed number of hops; names have
to be copied exactly as Wikidata writes them, capitals included, as smollerGraph's name index
matches them exactly; and the model asks once per question, where the [agent](agent.md) could ask
again.
