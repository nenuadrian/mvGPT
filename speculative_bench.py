import argparse
import json
import os
import statistics
import time

import torch

from bench import sync
from checkpoint import load_checkpoint
from hellaswag import RESULTS
from model import KVCache, pick_device

NEW = [1, 2, 3, 4, 5, 6, 8, 12, 16]  # tokens in a pass
CONTEXT = 100  # tokens already in the cache, about an agent's run: its prompt and what it has written


@torch.no_grad()
def pass_ms(model, device, new, repeats=40):
    """The median time of a pass of `new` tokens, after CONTEXT in the KV cache, in milliseconds."""
    times = []
    for _ in range(repeats):
        cache = KVCache(model.config.n_layer)
        model(torch.randint(0, 1000, (1, CONTEXT), device=device), cache=cache)
        x = torch.randint(0, 1000, (1, new), device=device)
        sync(device)
        start = time.perf_counter()
        model(x, cache=cache)
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    # The first passes on a device warm it up, so they're left out.
    return statistics.median(times[3:])


def main():
    parser = argparse.ArgumentParser(description="What a forward pass costs, by the number of tokens in it.")
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    devices = [pick_device(), "cpu"] if pick_device() != "cpu" else ["cpu"]
    results = {"context": CONTEXT, "devices": {}}
    for device in devices:
        model = load_checkpoint(args.checkpoint, device)[0].eval()
        times = {new: pass_ms(model, torch.device(device), new) for new in NEW}
        results["devices"][device] = {"ms": times, "per_token": {new: ms / new for new, ms in times.items()}}
        print(f"\n{device}: a pass after {CONTEXT} tokens\n{'tokens':>8}{'ms':>9}{'ms a token':>13}")
        for new, ms in times.items():
            print(f"{new:>8}{ms:>9.1f}{ms / new:>13.2f}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "speculative-bench.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
