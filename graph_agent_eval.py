import argparse
import json
import os
from collections import defaultdict

import agent_data
import graph_data
from agent import Agent
from agent_eval import attempt, summary
from checkpoint import load_checkpoint
from graph import SERVER
from graph_agent_data import as_role, load
from hellaswag import RESULTS
from model import pick_device

NEW_SETS = ["seen", "phrasing", "unknown"]
GRAPH_SETS = ["seen", "phrasing"]  # episode 24's held-out questions, asked of the graph role
OLD_SETS = ["math", "files", "files_new", "agent", "agent_new"]  # episode 21's held-out tasks


def facts(example):
    # The graph subagents' answers a task needs, in order: what follows each <<ask graph: ...>> in its trace.
    segments = example["segments"]
    return [segments[i + 1][0].removeprefix(" = ") for i, (text, learn) in enumerate(segments[:-1])
            if learn and "<<ask graph:" in text]


def before(name, set_name):
    # A set's right answers in an earlier episode's results, if they're here: episode 21's agent, episode 24's graph.
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["sets"].get(set_name, {}).get("right")


def main():
    parser = argparse.ArgumentParser(description="How well the agent asks the graph, and what it kept of the rest.")
    parser.add_argument("checkpoint")
    parser.add_argument("--limit", type=int, help="only the first examples of each set")
    parser.add_argument("--server", default=SERVER, help="where sg serve is listening")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    name = os.path.basename(os.path.dirname(args.checkpoint))
    results = {"model": name, "sets": {}}

    def agent_for(sandbox):
        return Agent(model, tokenizer, sandbox, server=args.server)

    print(f"the agent's new tasks\n{'set':<12}{'right':>8}{'facts right':>13}{'calls':>8}{'subagents':>11}{'tokens':>9}")
    by_kind = defaultdict(list)
    for set_name in NEW_SETS:
        examples = load(set_name)[: args.limit]
        rows = [attempt(agent_for, e) for e in examples]
        for e, r in zip(examples, rows):
            # Did the graph subagents bring back what the task needs? Then a wrong answer is the agent's own.
            r["facts_right"] = [res for call, res in r["asked"] if call.startswith("ask graph")] == facts(e)
            if set_name != "unknown":
                by_kind[r["kind"]].append(r["right"])
        s = summary(rows)
        s["facts_right"] = sum(r["facts_right"] for r in rows) / len(rows)
        s["by_kind"] = {k: sum(r["right"] for r in rows if r["kind"] == k) / sum(r["kind"] == k for r in rows)
                        for k in sorted({r["kind"] for r in rows})}
        s["wrong"] = [r["trace"] for r in rows if not r["right"]][:3]
        results["sets"][set_name] = s
        print(f"{set_name:<12}{s['right']:>8.1%}{s['facts_right']:>13.1%}{s['calls']:>8.1f}{s['subagents']:>11.1f}"
              f"{s['tokens']:>9.0f}", flush=True)
    results["by_kind"] = {k: sum(v) / len(v) for k, v in sorted(by_kind.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}
    print("right by kind, seen and phrasing: " + ", ".join(f"{k} {v:.0%}" for k, v in results["by_kind"].items()))

    # The graph role on its own, on the questions episode 24 measured its model on (graph_data.py's held-out sets).
    print(f"\nthe graph role, on episode 24's questions\n{'set':<12}{'right':>8}{'episode 24':>12}")
    for set_name in GRAPH_SETS:
        rows = [attempt(agent_for, as_role(e)) for e in graph_data.load(set_name)[: args.limit]]
        s = summary(rows)
        s["before"] = before("graph-graph.json", set_name)
        results["sets"][f"graph_{set_name}"] = s
        then = f"{s['before']:.1%}" if s["before"] is not None else "-"
        print(f"{set_name:<12}{s['right']:>8.1%}{then:>12}", flush=True)

    # Episode 21's tasks, which the new ones mustn't have cost: its held-out sets (agent_data.py's), as agent_eval.py
    # measures them.
    print(f"\nepisode 21's tasks\n{'set':<12}{'right':>8}{'episode 21':>12}")
    for set_name in OLD_SETS:
        rows = [attempt(agent_for, e) for e in agent_data.load(set_name)[: args.limit]]
        s = summary(rows)
        s["before"] = before("agent-agent.json", set_name)
        s["wrong"] = [r["trace"] for r in rows if not r["right"]][:3]
        results["sets"][f"episode21_{set_name}"] = s
        then = f"{s['before']:.1%}" if s["before"] is not None else "-"
        print(f"{set_name:<12}{s['right']:>8.1%}{then:>12}", flush=True)

    for set_name in NEW_SETS:
        wrong = results["sets"][set_name]["wrong"]
        if wrong:
            print(f"\na wrong one, {set_name}:\n" + wrong[0])
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"graph-agent-{name}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
