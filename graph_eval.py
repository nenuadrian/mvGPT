import argparse
import json
import os
import re
from collections import defaultdict

from checkpoint import load_checkpoint
from graph import CLOSE, OPEN, SERVER, generate_with_graph
from graph_data import load
from hellaswag import RESULTS
from model import pick_device
from tokenizer import EOT

GRAPH_SETS = ["seen", "phrasing", "missing"]
# A name copied with the space its first token has in the question: {name: ' Westland Petrel'}.
SPACED = re.compile(r"name: ['\"] ")


def expected(example):
    # The sentence the model should write after sg serve's answer, or after the question when no call is due.
    return example["segments"][-1][0].removeprefix(". ").replace(EOT, "").strip()


def score(example, text, calls):
    called = OPEN.strip() in text
    after = text.split(CLOSE, 1)[1] if CLOSE in text else text
    if example["kind"] == "numbers":
        return {"called": called, "right": not called and expected(example) in after}
    # A query is right if sg serve's answer to it is the right query's: another query that finds the same counts.
    return {"called": called, "query_right": bool(calls) and calls[0][1] == example["result"],
            "right": expected(example) in after, "spaced": bool(SPACED.search(text))}


def run(model, tokenizer, examples, use_tool, server):
    rows = []
    for e in examples:
        out, calls = generate_with_graph(model, tokenizer, tokenizer.encode(e["prompt"]), use_tool=use_tool,
                                         server=server)
        text = tokenizer.decode(out).strip()
        rows.append({**score(e, text, calls), "kind": e["kind"], "text": text})
    return rows


def rate(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description="How well a model asks smollerGraph, on the held-out questions.")
    parser.add_argument("checkpoint")
    parser.add_argument("--limit", type=int, help="only the first examples of each set")
    parser.add_argument("--server", default=SERVER, help="where sg serve is listening")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    name = os.path.basename(os.path.dirname(args.checkpoint))
    results = {"model": name, "sets": {}}
    answered = {}
    print(f"{'set':<10}{'calls':>8}{'query right':>13}{'answer right':>14}{'without the graph':>19}")
    for set_name in [*GRAPH_SETS, "numbers"]:
        examples = load(set_name)[: args.limit]
        rows = run(model, tokenizer, examples, True, args.server)
        answered[set_name] = rows
        m = {"examples": len(rows), "called": rate(rows, "called"), "right": rate(rows, "right")}
        if set_name in GRAPH_SETS:
            off = run(model, tokenizer, examples, False, args.server)
            m.update(query_right=rate(rows, "query_right"), right_without_graph=rate(off, "right"),
                     spaced=rate(rows, "spaced"))
            print(f"{set_name:<10}{m['called']:>8.1%}{m['query_right']:>13.1%}{m['right']:>14.1%}"
                  f"{m['right_without_graph']:>19.1%}", flush=True)
        else:
            print(f"{set_name:<10}{m['called']:>8.1%}{'':>13}{m['right']:>14.1%}", flush=True)
        m["samples"] = [r["text"] for r in rows[:3]]
        # A few it got wrong, with their questions, to see what goes wrong.
        m["wrong"] = [(e["prompt"].split("\n")[0][len("Question: "):], r["text"])
                      for e, r in zip(examples, rows) if not r["right"]][:3]
        results["sets"][set_name] = m

    # By kind, on the questions about species and groups the graph has: in the training's wordings and in new ones.
    by_kind = defaultdict(list)
    for set_name in ("seen", "phrasing"):
        for r in answered[set_name]:
            by_kind[r["kind"]].append(r["right"])
    results["by_kind"] = {kind: {"right": sum(v) / len(v), "examples": len(v)}
                          for kind, v in sorted(by_kind.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}
    print("\nright by kind: " + ", ".join(f"{kind} {k['right']:.0%}" for kind, k in results["by_kind"].items()))
    print("a space after the quote: " + ", ".join(f"{s} {results['sets'][s]['spaced']:.1%}" for s in GRAPH_SETS))
    print()
    for set_name in GRAPH_SETS:
        for q, text in results["sets"][set_name]["wrong"][:2]:
            print(f"{set_name:<10}{q}\n{'':<10}{text[:200]}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"graph-{name}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
