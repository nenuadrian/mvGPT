import argparse
import json
import math
import os
import re

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from data import DATA_DIR
from model import pick_device
from sft import build_examples, instruction, make_batch, parse
from tokenizer import EOT

PROMPT = "Once upon a time, there was a little"
INSTRUCTIONS = 50
MAX_TOKENS = 200


@torch.no_grad()
def story_metrics(model, data, windows=200, batch=25):
    model.eval()
    device = next(model.parameters()).device
    T = model.config.block_size
    total, correct, count = 0.0, 0, 0
    # The same windows for each model: back to back, from the start of the validation stories.
    for first in range(0, windows * T, batch * T):
        starts = range(first, first + batch * T, T)
        x = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in starts]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + T].astype(np.int64)) for i in starts]).to(device)
        logits, loss = model(x, y)
        total += loss.item() * y.numel()
        correct += (logits.argmax(-1) == y).sum().item()
        count += y.numel()
    return {"loss": total / count, "perplexity": math.exp(total / count), "accuracy": correct / count}


@torch.no_grad()
def next_tokens(model, tokenizer, prompt, k=5):
    model.eval()
    device = next(model.parameters()).device
    logits, _ = model(torch.tensor([tokenizer.encode(prompt)], device=device))
    probs = F.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, k)
    return {tokenizer.id_to_token[i]: p for p, i in zip(top.values.tolist(), top.indices.tolist())}


def load_instructions(path, n):
    with open(path, encoding="utf-8") as f:
        docs = [parse(doc) for doc in f.read().split(EOT)]
    # Instructions that ask for words, so we can check whether the story uses them.
    asks = [fields for fields in docs if "Words" in fields and "Story" in fields][:n]
    return [(instruction(fields), fields["Words"].split(", ")) for fields in asks]


@torch.no_grad()
def instruction_metrics(model, tokenizer, instructions, examples):
    model.eval()
    device = next(model.parameters()).device
    used, stopped, lengths = 0, 0, []
    for prompt, words in instructions:
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        out = model.generate(ids, MAX_TOKENS, temperature=0.8, top_k=50, eos_id=tokenizer.eot_id)
        story_ids = out[0, ids.size(1) :].tolist()
        story = tokenizer.decode(story_ids)
        # A word counts in its plural, or with -d, -ed or -ing: "play" in "played".
        used += sum(bool(re.search(rf"\b{re.escape(w)}(s|es|d|ed|ing)?\b", story, re.I)) for w in words)
        stopped += story_ids[-1] == tokenizer.eot_id
        lengths.append(len(story_ids))
    batches = range(0, len(examples), 25)
    loss = sum(model(*make_batch(examples[i : i + 25], tokenizer.pad_id, device))[1].item() for i in batches)
    return {
        "instruction_loss": loss / len(batches),
        "words_used": used / sum(len(words) for _, words in instructions),
        "stopped": stopped / len(instructions),
        "story_tokens": float(np.mean(lengths)),
    }


def name(path):
    # checkpoints/pretrain/model.pt is "pretrain", and checkpoints/pretrain/step-250.pt is "step-250".
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.basename(os.path.dirname(path)) if stem == "model" else stem


def main():
    parser = argparse.ArgumentParser(description="Measure trained models side by side, on held-out stories and on following instructions.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out", help="a folder to write the results to, as JSON")
    args = parser.parse_args()

    device = pick_device()
    valid = np.memmap(os.path.join(DATA_DIR, "valid.bin"), dtype=np.uint16, mode="r")
    instructions = load_instructions(os.path.join(DATA_DIR, "instruct_valid.txt"), INSTRUCTIONS)
    results, predictions = {}, {}
    for path in args.checkpoints:
        torch.manual_seed(1337)
        model, tokenizer, _ = load_checkpoint(path, device)
        examples = build_examples(os.path.join(DATA_DIR, "instruct_valid.txt"), tokenizer,
                                  model.config.block_size, limit=500)
        results[name(path)] = {**story_metrics(model, valid),
                               **instruction_metrics(model, tokenizer, instructions, examples)}
        predictions[name(path)] = next_tokens(model, tokenizer, PROMPT)

    names = list(results)
    print(f"{'':<18}" + "".join(f"{n:>12}" for n in names))
    for metric in results[names[0]]:
        cells = [results[n][metric] for n in names]
        percent = metric in ("accuracy", "words_used", "stopped")
        print(f"{metric:<18}" + "".join(f"{c:>12.1%}" if percent else f"{c:>12.2f}" for c in cells))
    for n in names:
        print(f"\n{n}: {PROMPT} ...")
        for token, p in predictions[n].items():
            print(f"  {token:<10}{p:6.1%} " + "#" * round(p * 50))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "models.json"), "w") as f:
            json.dump([{"model": n, **{k: round(v, 4) for k, v in results[n].items()}} for n in names], f, indent=1)
        for n in names:
            with open(os.path.join(args.out, f"next-{n}.json"), "w") as f:
                json.dump([{"token": t, "p": round(p, 4)} for t, p in predictions[n].items()], f, indent=1)


if __name__ == "__main__":
    main()
