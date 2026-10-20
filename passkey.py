import argparse
import json
import os
import random

import torch

from checkpoint import load_checkpoint
from hellaswag import RESULTS
from model import SCALINGS, KVCache, pick_device, stretch

# The pass key test of the position interpolation paper (its section 3.3): a five-digit number hidden in
# filler, asked for at the end, word for word.
INTRO = ("There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. "
         "I will quiz you about the important information there.\n")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
KEY = "The pass key is {key}. Remember it. {key} is the pass key. "
QUESTION = "What is the pass key? The pass key is"
DEPTHS = [0, 0.25, 0.5, 0.75, 1]  # where the key goes in the filler: 0 is the start, 1 right before the question
TRIALS = 10  # keys at each length and depth


def prompt(tokenizer, length, depth, key):
    # About length tokens: the filler repeated around the key, as many times as fit.
    per_filler = len(tokenizer.encode(FILLER * 10)) / 10
    fixed = len(tokenizer.encode(INTRO + KEY.format(key=key) + QUESTION))
    repeats = int((length - fixed) / per_filler) - 1
    before = round(repeats * depth)
    return tokenizer.encode(INTRO + FILLER * before + KEY.format(key=key) + FILLER * (repeats - before) + QUESTION)


@torch.no_grad()
def answer(model, ids, new_tokens=6):
    # Greedy, the likeliest token each time, through the KV cache.
    cache = KVCache(model.config.n_layer)
    logits, _ = model(ids, cache=cache)
    out = []
    for _ in range(new_tokens):
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        out.append(next_id.item())
        logits, _ = model(next_id, cache=cache)
    return out


def main():
    parser = argparse.ArgumentParser(description="Find the pass key: a number hidden in filler, past the window trained on.")
    parser.add_argument("checkpoint")
    parser.add_argument("--factor", type=int, default=4, help="prompts of one window up to this many")
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    name = os.path.basename(os.path.dirname(args.checkpoint))
    trained = model.config
    window = int(trained.block_size / trained.rope_factor)
    lengths = [window * n for n in range(1, args.factor + 1)]
    # As in long.py: each method on a model trained on one window, or a stretched model as it is.
    configs = {s or "none": stretch(trained, s, args.factor) for s in (None, *SCALINGS)}
    if trained.rope_scaling:
        configs = {trained.rope_scaling: trained}

    results = {"model": name, "window": window, "lengths": lengths, "depths": DEPTHS, "trials": TRIALS, "methods": {}}
    print(f"pass keys found, of {TRIALS * len(DEPTHS)} at each length: {TRIALS} at each of {len(DEPTHS)} depths\n")
    print(f"{'method':<10}" + "".join(f"{n:>9,}" for n in lengths))
    for method, config in configs.items():
        model.config = config
        rng = random.Random(0)  # the same keys for each method
        grid = []
        for length in lengths:
            row = []
            for depth in DEPTHS:
                found = 0
                for _ in range(TRIALS):
                    key = str(rng.randint(10_000, 99_999))
                    ids = torch.tensor([prompt(tokenizer, length, depth, key)], device=args.device)
                    found += key in tokenizer.decode(answer(model, ids))
                row.append(found)
            grid.append(row)
        found = [sum(row) / (TRIALS * len(DEPTHS)) for row in grid]
        results["methods"][method] = {"found": grid, "by_length": found}
        print(f"{method:<10}" + "".join(f"{x:>9.0%}" for x in found), flush=True)

    os.makedirs(os.path.join(RESULTS, "passkey"), exist_ok=True)
    with open(os.path.join(RESULTS, "passkey", f"{name}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
