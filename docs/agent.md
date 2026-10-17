# An agent, with subagents

The model [pretrained on the web](web.md) learned to [call a calculator](tools.md). Here it becomes
an agent: one model, fine-tuned once into three roles. The agent works with files and hands the
rest of a task to subagents, which are the same model in a fresh context.

## One model, three roles

The prompt names a role and its tools:

```
Role: agent
Tools: ask math, ask files, write, list
Task: Add up the numbers in jan.txt and feb.txt.
Answer: <<ask files: Add up the numbers in jan.txt.>> = 465. <<ask files: Add up the numbers in feb.txt.>> = 309. <<ask math: What is 465 + 309?>> = 774. The total is 774.
```

| Role | Tools |
| --- | --- |
| agent | `ask math`, `ask files`, `write`, `list` |
| math | `calc` |
| files | `read`, `write`, `list`, `calc` |

Calls are plain text, as in `tools.py`. When one closes, `agent.py` checks that the role has the
tool and runs it. `ask` starts a subagent: the model again, in the role asked for, with a prompt of
its own and a KV cache of its own. Its final answer alone comes back as the result: the file's
numbers stay out of the agent's context. This is what [ReAct](https://arxiv.org/abs/2210.03629)
calls acting, and what Anthropic's
[multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
does with subagents, at a far smaller scale.

## Limits, in the program

`Sandbox` is the one folder the model can see. A file name has to match `[a-z0-9_-]{1,24}\.txt`,
so no name has a slash in it, and no path leads out. A session holds at most 20 files of 400
characters, and content with `<` or `>`, which would look like a call, is refused. A role can't use
another role's tools, a subagent can't start subagents, and a context stops after 10 calls. Each
refusal comes back to the model as a result starting with `error:`.

```bash
python -m pytest tests/test_agent.py
python agent.py checkpoints/agent/model.pt "Add up the numbers in jan.txt and feb.txt." --files '{"jan.txt": "120, 45, 300", "feb.txt": "18, 7, 220, 64"}'
python agent.py checkpoints/agent/model.pt "Add up the numbers in jan.txt and feb.txt." --role files   # in one context
python agent_app.py   # the chat, at http://127.0.0.1:7861
```

## Sixty thousand traces

`agent_data.py` generates them with a seed: 15,017 for math (episode 20's questions, and sums of 3
or 4 numbers), 24,031 for files (save, read, list, add up a file, add up and save, copy, missing
files, and several files in one context), 17,960 for the agent (sums and word problems saved to
files, two at once, and totals of several files, delegated), and 2,992 web pages. Each trace marks
the prompt and the tools' results as given, and the calls and sentences as learned. The held-out
sets use new numbers, and the `_new` ones new wordings and file names.

`agent_sft.py` fine-tunes `pretrain-web` on all of it with `tool_sft.py`'s mask: 30 minutes on an
Apple M-series laptop, 1,899 steps, 58,451 examples, about one pass. Batches hold examples of about
the same length, up to 32 and about 6,144 tokens, and the MPS cache is emptied every 50 steps: the
first run, with batches of 32 whatever their length, left 31 GB in it and ran out of memory.

## Results

`agent_eval.py` runs each held-out task in a fresh folder and checks the files left in it and the
answer:

| Held-out set | Right | Agent's context |
| --- | --- | --- |
| math | 100% | 48 tokens |
| files | 100% | 129 |
| agent | 100% | 100 |
| files, new wordings and names | 49% | 166 |
| agent, new wordings and names | 23% | 99 |

On new wordings, the names are copied well; the plans go wrong. "Store … in …" is read as a read,
"Calculate a and b and store them" does one of the two, and a held-out word problem adds where it
should subtract.

### Subagents against one context

Four files of 10 to 15 numbers to add up, 100 times: the agent hands each to a files subagent and
the totals to a math one, or the files role reads all four into one context and adds 40 to 60
numbers in one call.

| | Right | Its context | Largest context | All contexts |
| --- | --- | --- | --- | --- |
| With subagents | 95% | 159 tokens | 167 | 719 |
| In one context | 80% | 377 | 687 | 377 |

One context goes wrong in the long copy, skipping part of the numbers on the way into the
calculator. Subagents keep each context small, for about twice the tokens in all.

## What's real, and what's scaled down

The loop is real: calls, results read back, and subagents in fresh contexts that hand back their
answers, with the limits in the program. The tasks are not: a few kinds, from templates, and the
agent follows the plans it saw in training rather than making new ones, which is why new wordings
trip it up.
