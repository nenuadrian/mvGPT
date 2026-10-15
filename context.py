import argparse
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from bench import sync
from checkpoint import load_checkpoint
from data import DATA_DIR
from model import KVCache, pick_device

LENGTH = 768  # three context windows of held-out text
STREAMS = 8


@torch.no_grad()
def window_losses(model, ids):
    # Each token past the first window, predicted from the 256 tokens before it, run in full each time,
    # one window after another, as writing would have to.
    W = model.config.block_size
    losses = []
    for end in range(W, ids.size(1)):
        logits, _ = model(ids[:, end - W : end])
        losses.append(F.cross_entropy(logits[:, -1], ids[:, end]))
    return torch.stack(losses)


@torch.no_grad()
def cache_losses(model, ids):
    # A token at a time through the cache, which slides once it holds a window's worth of tokens.
    W = model.config.block_size
    cache, losses = KVCache(model.config.n_layer), []
    for t in range(ids.size(1) - 1):
        if len(cache) == W:
            cache.slide(W - 1)
        logits, _ = model(ids[:, t : t + 1], cache=cache)
        if t + 1 >= W:
            losses.append(F.cross_entropy(logits[:, -1], ids[:, t + 1]))
    return torch.stack(losses)


@torch.no_grad()
def full_losses(model, ids):
    # All the tokens at once, each attending to all the ones before it, past the window it was trained on.
    logits, _ = model(ids[:, :-1])
    return F.cross_entropy(logits[0, model.config.block_size - 1 :], ids[0, model.config.block_size :], reduction="none")


def timed(fn, model, ids):
    sync(ids.device)
    start = time.perf_counter()
    losses = fn(model, ids)
    sync(ids.device)
    return losses, (time.perf_counter() - start) * 1000 / len(losses)


def main():
    parser = argparse.ArgumentParser(description="Measure models past their context window.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    valid = np.memmap(os.path.join(DATA_DIR, "valid.bin"), dtype=np.uint16, mode="r")
    streams = [torch.from_numpy(valid[i * LENGTH : (i + 1) * LENGTH].astype(np.int64))[None].to(args.device)
               for i in range(STREAMS)]
    print(f"loss on tokens 256 to {LENGTH - 1} of {STREAMS} held-out streams, and the time per token\n")
    print(f"{'model':<16}{'method':<28}{'loss':>8}{'ms':>8}")
    for path in args.checkpoints:
        model, _, _ = load_checkpoint(path, args.device)
        model.eval()
        name = os.path.basename(os.path.dirname(path))
        methods = {"sliding window": window_losses}
        # Learned positions stop at 256, so the cache can't slide, and the model can't see further.
        if model.config.rope:
            methods.update({"sliding cache": cache_losses, "all of the context": full_losses})
        for method, fn in methods.items():
            results = [timed(fn, model, ids) for ids in streams]
            loss = torch.cat([losses for losses, _ in results]).mean().item()
            ms = np.mean([ms for _, ms in results])
            print(f"{name:<16}{method:<28}{loss:>8.3f}{ms:>8.2f}")


if __name__ == "__main__":
    main()
