import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict

from agent import NUMBER, Agent, Sandbox, show
from agent_data import load
from checkpoint import load_checkpoint
from hellaswag import RESULTS
from model import pick_device

SETS = ["math", "files", "files_new", "agent", "agent_new"]


def all_runs(run):
    # A run and its subagents' runs, each its own context.
    return [run] + [r for _, _, sub in run.calls if sub is not None for r in all_runs(sub)]


def attempt(agent_for, example, role=None):
    """Runs one task in a folder of its own, with the files it starts with, and checks what it left: the files,
    all of them and nothing more, and the answer in the final sentence."""
    folder = tempfile.mkdtemp()
    try:
        sandbox = Sandbox(folder, example["files"])
        run = agent_for(sandbox).run(role or example["role"], example["task"])
        expect = example["expect"]
        files_right = sandbox.names() == sorted(expect["files"]) and \
            all(sandbox.read(n) == c for n, c in expect["files"].items())
        answer = expect.get("answer")
        if answer is None:
            answer_right = True
        elif NUMBER.fullmatch(answer):
            answer_right = answer in NUMBER.findall(run.final)
        else:
            answer_right = answer in run.final
        runs = all_runs(run)
        return {"kind": example["kind"], "right": files_right and answer_right, "calls": len(run.calls),
                "subagents": len(runs) - 1, "tokens": run.tokens, "largest": max(r.tokens for r in runs),
                "all_tokens": sum(r.tokens for r in runs), "trace": show(run),
                "asked": [(call, result) for call, result, _ in run.calls]}
    finally:
        shutil.rmtree(folder)


def summary(rows):
    n = len(rows)
    return {"examples": n, "right": sum(r["right"] for r in rows) / n, "calls": sum(r["calls"] for r in rows) / n,
            "subagents": sum(r["subagents"] for r in rows) / n, "tokens": sum(r["tokens"] for r in rows) / n,
            "largest": max(r["largest"] for r in rows), "all_tokens": sum(r["all_tokens"] for r in rows) / n}


def main():
    parser = argparse.ArgumentParser(description="How often the agent does its held-out tasks right, with what it costs.")
    parser.add_argument("checkpoint")
    parser.add_argument("--limit", type=int, help="only the first examples of each set")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())

    def agent_for(sandbox):
        return Agent(model, tokenizer, sandbox)

    results = {"model": os.path.basename(os.path.dirname(args.checkpoint)), "sets": {}}
    print(f"{'set':<12}{'right':>8}{'calls':>8}{'subagents':>11}{'tokens':>9}")
    for name in SETS:
        rows = [attempt(agent_for, e) for e in load(name)[: args.limit]]
        s = summary(rows)
        by_kind = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r["right"])
        s["by_kind"] = {k: sum(v) / len(v) for k, v in sorted(by_kind.items())}
        s["wrong"] = [r["trace"] for r in rows if not r["right"]][:3]
        results["sets"][name] = s
        print(f"{name:<12}{s['right']:>8.1%}{s['calls']:>8.1f}{s['subagents']:>11.1f}{s['tokens']:>9.0f}", flush=True)

    # One long task two ways: the agent hands each file to a subagent, or one files context reads all four.
    print(f"\nadding up four files of 10 to 15 numbers each:\n{'':<20}{'right':>8}{'its context':>13}"
          f"{'largest':>9}{'all contexts':>14}")
    context = load("context")[: args.limit]
    for label, role in (("with subagents", "agent"), ("in one context", "files")):
        rows = [attempt(agent_for, e, role) for e in context]
        s = summary(rows)
        s["wrong"] = [r["trace"] for r in rows if not r["right"]][:3]
        results["sets"][f"context_{role}"] = s
        print(f"{label:<20}{s['right']:>8.1%}{s['tokens']:>13.0f}{s['largest']:>9}{s['all_tokens']:>14.0f}", flush=True)
    for role in ("agent", "files"):
        wrong = results["sets"][f"context_{role}"]["wrong"]
        if wrong:
            print(f"\na wrong one, {role}:\n" + wrong[0])
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"agent-{results['model']}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
