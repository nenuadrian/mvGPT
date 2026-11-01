import argparse
import json
import os
import random
import statistics
import time

import torch
from torch.nn import functional as F

from bench import sync
from checkpoint import load_checkpoint
from hellaswag import RESULTS
from long import full_losses, long_pages
from model import KVCache, pick_device
from passkey import DEPTHS, TRIALS, answer, prompt

BINS = 16  # each window's positions in this many groups


def kind(model):
    c = model.config
    if c.mixer != "hyena":
        return "attention"
    return "hybrid" if c.attention_layers else "Hyena"


@torch.no_grad()
def loss_by_position(model, tokenizer, device, pages):
    # Each token of long held-out pages predicted from all the ones before it in the window, averaged by position:
    # a model that uses what it has read gets better the further into the window it is.
    W = model.config.block_size
    losses = torch.stack([full_losses(model, ids) for ids in long_pages(tokenizer, W, pages, device)])
    bins = losses.view(len(losses), BINS, -1).mean(dim=(0, 2))
    return {"loss": round(losses.mean().item(), 4), "bins": [round(x, 4) for x in bins.tolist()],
            "bin": W // BINS}


@torch.no_grad()
def copying(model, tokenizer, device, n=32):
    # Random tokens, then the same ones again: nothing in the first half can be guessed, everything in the second can,
    # by finding the token in the first copy and taking the one after it. In-context copying, as induction heads do.
    half = (model.config.block_size - 1) // 2
    g = torch.Generator().manual_seed(0)
    # Any token but the padding and the end of text, the two last ids.
    random_ids = torch.randint(256, tokenizer.pad_id, (n, half), generator=g)
    ids = torch.cat([torch.full((n, 1), tokenizer.eot_id), random_ids, random_ids], dim=1).to(device)
    # One sequence at a time: 32 windows of 4,096 tokens make more logits than the Mac's GPU takes in one tensor.
    losses, right = [], []
    for row in ids:
        logits, _ = model(row[None, :-1])
        losses.append(F.cross_entropy(logits[0], row[1:], reduction="none"))
        right.append(logits[0].argmax(dim=-1) == row[1:])
    losses, right = torch.stack(losses), torch.stack(right)
    return {"tokens": half, "first": round(losses[:, :half].mean().item(), 4),
            "second": round(losses[:, half + 1 :].mean().item(), 4),
            "copied": round(right[:, half + 1 :].float().mean().item(), 4)}


@torch.no_grad()
def pass_keys(model, tokenizer, device):
    # passkey.py's test, at a quarter of the window up to all of it: a Hyena filter is as long as the window, no more.
    W = model.config.block_size
    rng = random.Random(0)
    found = {}
    for length in (W // 4, W // 2, 3 * W // 4, W - 8):
        hits = 0
        for depth in DEPTHS:
            for _ in range(TRIALS):
                key = str(rng.randint(10_000, 99_999))
                ids = torch.tensor([prompt(tokenizer, length, depth, key)], device=device)
                hits += key in tokenizer.decode(answer(model, ids))
        found[length] = hits / (TRIALS * len(DEPTHS))
    return found


@torch.no_grad()
def writing(model, device, new_tokens=64):
    # Milliseconds a token, writing through the cache, after prompts of a quarter of the window up to most of it.
    W = model.config.block_size
    out = {}
    for length in (W // 4, W // 2, W - new_tokens):
        ids = torch.randint(256, model.config.vocab_size, (1, length), device=device)
        cache = KVCache(model.config.n_layer)
        logits, _ = model(ids, cache=cache)
        times = []
        for _ in range(new_tokens):
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            sync(ids.device)
            start = time.perf_counter()
            logits, _ = model(next_id, cache=cache)
            sync(ids.device)
            times.append((time.perf_counter() - start) * 1000)
        out[length] = round(statistics.median(times), 3)
    return out


def main():
    parser = argparse.ArgumentParser(description="Attention, Hyena and hybrid models side by side: loss by position, "
                                     "copying, pass keys and writing speed, in each model's window.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--pages", type=int, default=64)
    parser.add_argument("--skip", nargs="*", default=[], choices=["loss", "copy", "passkey", "writing"])
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--out", help="a JSON file for the results (default results/hyena/<first model>.json)")
    args = parser.parse_args()

    results = {}
    for path in args.checkpoints:
        model, tokenizer, _ = load_checkpoint(path, args.device)
        model.eval()
        name = os.path.basename(os.path.dirname(path))
        r = {"kind": kind(model), "window": model.config.block_size, "params": model.num_params()}
        print(f"{name}: {r['kind']}, a window of {r['window']:,} tokens, {r['params']:,} parameters", flush=True)
        if "loss" not in args.skip:
            r["loss"] = loss_by_position(model, tokenizer, args.device, args.pages)
            bins = r["loss"]["bins"]
            print(f"  held-out loss {r['loss']['loss']:.3f}: {bins[0]:.3f} in the first {r['loss']['bin']:,} "
                  f"tokens, {bins[-1]:.3f} in the last", flush=True)
        if "copy" not in args.skip:
            r["copy"] = copying(model, tokenizer, args.device)
            c = r["copy"]
            print(f"  {c['tokens']:,} random tokens twice: loss {c['first']:.2f} the first time, {c['second']:.2f} "
                  f"the second, {c['copied']:.0%} copied", flush=True)
        if "passkey" not in args.skip:
            r["passkey"] = pass_keys(model, tokenizer, args.device)
            print("  pass keys found: " + ", ".join(f"{v:.0%} in {k:,} tokens" for k, v in r["passkey"].items()),
                  flush=True)
        if "writing" not in args.skip:
            r["writing"] = writing(model, args.device)
            print("  writing: " + ", ".join(f"{v:.1f} ms a token after {k:,}" for k, v in r["writing"].items()),
                  flush=True)
        results[name] = r

    out = args.out or os.path.join(RESULTS, "hyena", f"{os.path.basename(os.path.dirname(args.checkpoints[0]))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
