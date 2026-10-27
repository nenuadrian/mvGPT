import argparse
import json
import os
import re
import time

import graph_data
import graph_agent_data
from agent import Agent
from agent_eval import attempt, summary
from checkpoint import load_checkpoint
from constrain import graph_names, task_words
from graph import SERVER
from graph_agent_eval import facts
from hellaswag import RESULTS
from model import pick_device

GRAPH_SETS = ["seen", "phrasing", "missing"]  # episode 24's held-out questions, asked of the graph role
NEW_SETS = ["seen", "phrasing", "unknown"]  # episode 25's held-out tasks, asked of the agent
# A name in a query, in single quotes, or in double ones when it has an apostrophe: {name: 'Okapi'}.
NAME = re.compile(r"\{name: (?:'([^']*)'|\"([^\"]*)\")\}")


def measure(agent_for, examples, nodes):
    """Each example's attempt, summed up, with the seconds an example took, and how often a query named something the
    graph hasn't got: a name copied wrong, or an animal that isn't in it."""
    start = time.perf_counter()
    rows = [attempt(agent_for, e) for e in examples]
    s = summary(rows)
    s["seconds"] = (time.perf_counter() - start) / len(rows)
    s["off_graph"] = sum(any((a or b) not in nodes for a, b in NAME.findall(r["trace"])) for r in rows) / len(rows)
    return s, rows


def main():
    parser = argparse.ArgumentParser(description="What keeping the graph role's names to a list does to its answers.")
    parser.add_argument("checkpoint")
    parser.add_argument("facts", help="smollerGraph's facts file, which sg serve has loaded")
    parser.add_argument("--limit", type=int, help="only the first examples of each set")
    parser.add_argument("--server", default=SERVER, help="where sg serve is listening")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    name = os.path.basename(os.path.dirname(args.checkpoint))
    nodes = graph_data.read_facts(args.facts)[0]
    # The queries written freely, with the graph's names, and with names copied from the task.
    modes = {"free": None, "graph": graph_names(tokenizer, args.facts), "task": task_words(tokenizer)}
    results = {"model": name, "graph_role": {}, "agent": {}}

    def agent_for(constrain):
        return lambda sandbox: Agent(model, tokenizer, sandbox, server=args.server, constrain=constrain)

    print(f"the graph role, on episode 24's questions\n{'set':<12}" + "".join(f"{m:>9}" for m in modes)
          + "    named off the graph")
    for set_name in GRAPH_SETS:
        examples = [graph_agent_data.as_role(e) for e in graph_data.load(set_name)[: args.limit]]
        row = {}
        for mode, constrain in modes.items():
            row[mode], rows = measure(agent_for(constrain), examples, nodes)
            row[mode]["wrong"] = [r["trace"] for r in rows if not r["right"]][:3]
        results["graph_role"][set_name] = row
        off = " / ".join(f"{row[m]['off_graph']:.0%}" for m in modes)
        print(f"{set_name:<12}" + "".join(f"{row[m]['right']:>9.1%}" for m in modes) + f"    {off}", flush=True)

    # The agent's own tasks, with its graph subagents writing freely, then copying their names from their tasks.
    agent_modes = ["free", "task"]
    print(f"\nthe agent, on episode 25's tasks\n{'set':<12}" + "".join(f"{m:>9}" for m in agent_modes)
          + "    facts right")
    for set_name in NEW_SETS:
        examples = graph_agent_data.load(set_name)[: args.limit]
        row = {}
        for mode in agent_modes:
            row[mode], rows = measure(agent_for(modes[mode]), examples, nodes)
            # Did the graph subagents bring back what the task needs?
            row[mode]["facts_right"] = sum([res for call, res in r["asked"] if call.startswith("ask graph")] == facts(e)
                                           for e, r in zip(examples, rows)) / len(rows)
            row[mode]["wrong"] = [r["trace"] for r in rows if not r["right"]][:3]
        results["agent"][set_name] = row
        right = " / ".join(f"{row[m]['facts_right']:.0%}" for m in agent_modes)
        print(f"{set_name:<12}" + "".join(f"{row[m]['right']:>9.1%}" for m in agent_modes) + f"    {right}", flush=True)

    # The time a question takes, which is the model's steps, the queries and the masks.
    results["seconds"] = {m: sum(results["graph_role"][s][m]["seconds"] for s in GRAPH_SETS) / len(GRAPH_SETS)
                          for m in modes}
    print("\nseconds a question, the graph role: " + ", ".join(f"{m} {v:.2f}" for m, v in results["seconds"].items()))
    # With the graph's names, an animal the graph hasn't got becomes one it has.
    print("\nwith the graph's names, a missing animal:\n" + results["graph_role"]["missing"]["graph"]["wrong"][0])
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"constrain-{name}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
