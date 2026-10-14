"""Measures trained models on held-out stories, side by side.

    python evaluate.py checkpoints/pretrain/step-0.pt checkpoints/pretrain/model.pt
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from data import DATA_DIR
from model import pick_device

PROMPT = "Once upon a time, there was a little"


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


def name(path):
    # checkpoints/pretrain/model.pt is "pretrain", and checkpoints/pretrain/step-250.pt is "step-250".
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.basename(os.path.dirname(path)) if stem == "model" else stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out", help="a folder to write the results to, as JSON")
    args = parser.parse_args()

    device = pick_device()
    valid = np.memmap(os.path.join(DATA_DIR, "valid.bin"), dtype=np.uint16, mode="r")
    results, predictions = {}, {}
    for path in args.checkpoints:
        torch.manual_seed(1337)
        model, tokenizer, _ = load_checkpoint(path, device)
        results[name(path)] = story_metrics(model, valid)
        predictions[name(path)] = next_tokens(model, tokenizer, PROMPT)

    names = list(results)
    print(f"{'':<14}" + "".join(f"{n:>12}" for n in names))
    for metric in results[names[0]]:
        cells = [results[n][metric] for n in names]
        print(f"{metric:<14}" + "".join(f"{c:>12.1%}" if metric == "accuracy" else f"{c:>12.2f}" for c in cells))
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
