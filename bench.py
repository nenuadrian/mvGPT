import argparse
import json
import os
import statistics
import time

import torch

from checkpoint import load_checkpoint
from model import KVCache, pick_device

PROMPT = "Once upon a time"
STORY = "Once upon a time, there was a little girl named Lily. She loved to play in the park with her dog."


def sync(device):
    # A GPU runs what it's given in the background: wait for it to finish before reading the clock.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def largest_difference(model, ids):
    # The logits for all the tokens at once, and for one token at a time through the cache.
    full, _ = model(ids)
    cache = KVCache(model.config.n_layer)
    steps = [model(ids[:, t : t + 1], cache=cache)[0] for t in range(ids.size(1))]
    return (full - torch.cat(steps, dim=1)).abs().max().item()


@torch.no_grad()
def step_times(model, prompt, use_cache):
    device = prompt.device
    cache = KVCache(model.config.n_layer) if use_cache else None
    ids, x, times = prompt, prompt, []
    while ids.size(1) < model.config.block_size:
        sync(device)
        start = time.perf_counter()
        logits, _ = model(x, cache=cache)
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
        next_id = logits[:, -1:].argmax(-1)
        ids = torch.cat([ids, next_id], dim=1)
        # With the cache, the next step runs the new token alone, and without it, all of them.
        x = next_id if use_cache else ids
    return times


@torch.no_grad()
def tokens_per_second(model, prompt, batch, use_cache, tokens=200):
    device = prompt.device
    ids = prompt.repeat(batch, 1)
    sync(device)
    start = time.perf_counter()
    model.generate(ids, tokens, temperature=0.8, top_k=50, use_cache=use_cache)
    sync(device)
    return batch * tokens / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description="Check the KV cache, and measure how fast the model writes with and without it.")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--out", help="a folder to write the timings to, as JSON")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    prompt = torch.tensor([tokenizer.encode(PROMPT)], device=args.device)

    difference = largest_difference(model, torch.tensor([tokenizer.encode(STORY)], device=args.device))
    print(f"largest difference in the logits: {difference:.1e}")
    stories = []
    for use_cache in (False, True):
        torch.manual_seed(1337)
        stories.append(model.generate(prompt, 200, temperature=0.8, top_k=50, use_cache=use_cache))
    match = torch.equal(*stories)
    print(f"stories match: {match}")

    # Five runs each, and the median at each length, as a single run is noisy.
    lengths = range(prompt.size(1), model.config.block_size)
    without = [statistics.median(t) for t in zip(*[step_times(model, prompt, False) for _ in range(5)])]
    with_cache = [statistics.median(t) for t in zip(*[step_times(model, prompt, True) for _ in range(5)])]
    print(f"\n{'tokens so far':<16}{'without':>10}{'with':>10}")
    for i in range(0, len(lengths), 50):
        print(f"{lengths[i]:<16}{without[i]:>8.2f}ms{with_cache[i]:>8.2f}ms")

    print(f"\n{'batch':<16}{'without':>13}{'with':>13}{'speed-up':>10}")
    speeds = []
    for batch in (1, 8):
        for use_cache in (False, True):
            tokens_per_second(model, prompt, batch, use_cache, tokens=20)  # a short run, to warm up
        slow = tokens_per_second(model, prompt, batch, False)
        fast = tokens_per_second(model, prompt, batch, True)
        speeds.append({"batch": batch, "without": round(slow), "with": round(fast), "speedup": round(fast / slow, 2)})
        print(f"{batch:<16}{slow:>7.0f} tok/s{fast:>7.0f} tok/s{fast / slow:>9.1f}x")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        steps = [{"tokens": n, "without_ms": round(a, 3), "with_ms": round(b, 3)}
                 for n, a, b in zip(lengths, without, with_cache)]
        with open(os.path.join(args.out, f"bench-{args.device}.json"), "w") as f:
            json.dump({"device": args.device, "difference": difference, "match": match,
                       "steps": steps, "speeds": speeds}, f, indent=1)


if __name__ == "__main__":
    main()
