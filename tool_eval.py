import argparse
import json
import os
import re
from collections import defaultdict

from checkpoint import load_checkpoint
from hellaswag import RESULTS
from model import pick_device
from tool_data import load
from tools import CLOSE, OPEN, calculate, generate_with_tools

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
CALL_SETS = ["seen", "phrasing", "word"]


def after_call(text):
    """The answer the model writes in words, after its call and the result: "There are 888 pencils." With no
    call, all of it. The result itself doesn't count, whether the calculator wrote it or the model."""
    if CLOSE not in text:
        return text
    rest = text.split(CLOSE, 1)[1]
    return re.sub(r"^\s*=\s*-?[\d.]*", "", rest)


def score(example, text, calls):
    called = OPEN.strip() in text
    numbers = NUMBER.findall(after_call(text))
    if example["kind"] == "web":
        return {"called": called}
    if example["kind"] == "numbers":
        return {"called": called, "right": not called and bool(numbers) and numbers[0] == example["answer"]}
    expression = calls[0][0] if calls else None
    return {"called": called, "call_right": expression is not None and calculate(expression) == example["answer"],
            "right": example["answer"] in numbers}


def run(model, tokenizer, examples, use_tool, shots=""):
    rows = []
    for e in examples:
        ids = tokenizer.encode(shots + e["prompt"])
        out, calls = generate_with_tools(model, tokenizer, ids, max_new_tokens=48, use_tool=use_tool)
        # With examples in front, a pretrained model carries on with another question: the answer ends before it.
        text = tokenizer.decode(out).split("\nQuestion:")[0].split("\n\n")[0]
        rows.append({**score(e, text, calls), "digits": e.get("digits"), "text": text})
    return rows


def rate(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description="How well a model calls the calculator, on the held-out questions.")
    parser.add_argument("checkpoint")
    parser.add_argument("--shots", action="store_true", help="three worked examples in front of each question, for a "
                                                             "model that wasn't fine-tuned to call")
    parser.add_argument("--limit", type=int, help="only the first examples of each set")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    shots = ""
    if args.shots:
        train = load("train")
        picks = [next(e for e in train if e["kind"] == kind) for kind in ("arithmetic", "word", "numbers")]
        shots = "".join("".join(text for text, _ in e["segments"]).replace("<|endoftext|>", "") + "\n\n" for e in picks)
    name = os.path.basename(os.path.dirname(args.checkpoint)) + ("-shots" if args.shots else "")
    results = {"model": name, "sets": {}}
    print(f"{'set':<10}{'calls':>8}{'call right':>12}{'answer right':>14}{'without the tool':>18}")
    for set_name in [*CALL_SETS, "numbers", "web"]:
        examples = load(set_name)[: args.limit]
        rows = run(model, tokenizer, examples, True, shots)
        m = {"examples": len(rows), "called": rate(rows, "called")}
        if set_name in CALL_SETS:
            off = run(model, tokenizer, examples, False, shots)
            m.update(call_right=rate(rows, "call_right"), right=rate(rows, "right"), right_without_tool=rate(off, "right"))
            # By the length of the longest number in the question.
            by = defaultdict(lambda: [[], []])
            for on, no in zip(rows, off):
                by[on["digits"]][0].append(on["right"])
                by[on["digits"]][1].append(no["right"])
            m["by_digits"] = [{"digits": d, "right": sum(a) / len(a), "right_without_tool": sum(b) / len(b),
                               "examples": len(a)} for d, (a, b) in sorted(by.items())]
            print(f"{set_name:<10}{m['called']:>8.1%}{m['call_right']:>12.1%}{m['right']:>14.1%}"
                  f"{m['right_without_tool']:>18.1%}", flush=True)
        elif set_name == "numbers":
            m["right"] = rate(rows, "right")
            print(f"{set_name:<10}{m['called']:>8.1%}{'':>12}{m['right']:>14.1%}")
        else:
            print(f"{set_name:<10}{m['called']:>8.1%}")
        m["samples"] = [r["text"] for r in rows[:3]]
        # A few of the calls it got wrong, with their questions, to see what goes wrong.
        m["wrong"] = [(e["prompt"].split("\n")[0][len("Question: "):], r["text"].split(">>")[0].strip(" <"))
                      for e, r in zip(examples, rows) if set_name in CALL_SETS and not r["call_right"]][:3]
        results["sets"][set_name] = m
    print()
    for set_name in CALL_SETS:
        for q, call in results["sets"][set_name]["wrong"][:2]:
            print(f"{set_name:<10}{q}\n{'':<10}{call}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"tools-{name}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
