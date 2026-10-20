import argparse
import json
import os

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from data import DATA_DIR
from hellaswag import RESULTS
from model import SCALINGS, pick_device, stretch

FACTOR = 4  # pages four windows long, and a model trained on one window stretched to fit them
PAGES = 64
BIN = 256  # positions averaged together for each point of the results


def long_spans(tokenizer, length, split="valid"):
    # Where each web page of more than length tokens starts and ends: each one ends with end of text.
    ids = np.memmap(os.path.join(DATA_DIR, f"{split}-web.bin"), dtype=np.uint16, mode="r")
    ends = np.flatnonzero(ids == tokenizer.eot_id)
    starts = np.concatenate([[0], ends[:-1] + 1])
    keep = ends - starts > length
    return ids, starts[keep], ends[keep]


def long_pages(tokenizer, length, n, device):
    # The first length + 1 tokens of the first n long held-out pages.
    ids, starts, _ = long_spans(tokenizer, length)
    return [torch.from_numpy(ids[s : s + length + 1].astype(np.int64))[None].to(device) for s in starts[:n]]


@torch.no_grad()
def full_losses(model, ids):
    # Every token predicted from all the ones before it on the page, in one pass.
    logits, _ = model(ids[:, :-1])
    return F.cross_entropy(logits[0], ids[0, 1:], reduction="none")


@torch.no_grad()
def window_losses(model, ids, window):
    # Only as many tokens as the window trained on: windows a quarter of one apart, each predicting its last
    # quarter, so each token past the first window sees from three quarters of a window to a whole one.
    logits, _ = model(ids[:, :window])
    losses = [F.cross_entropy(logits[0], ids[0, 1 : window + 1], reduction="none")]
    step = window // 4
    for end in range(window + step, ids.size(1), step):
        logits, _ = model(ids[:, end - window : end])
        losses.append(F.cross_entropy(logits[0, -step:], ids[0, end - step + 1 : end + 1], reduction="none"))
    return torch.cat(losses)


def main():
    parser = argparse.ArgumentParser(description="Loss by position on long held-out pages, past the window trained on.")
    parser.add_argument("checkpoint")
    parser.add_argument("--factor", type=float, default=FACTOR, help="pages this many windows long")
    parser.add_argument("--pages", type=int, default=PAGES)
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    name = os.path.basename(os.path.dirname(args.checkpoint))
    trained = model.config
    window = int(trained.block_size / trained.rope_factor)  # the window it was trained on
    length = int(window * args.factor)
    pages = long_pages(tokenizer, length, args.pages, args.device)

    # The same weights each time, read with another config: a model trained on one window is stretched by
    # each method, and one that was stretched and trained is measured as it is.
    configs = {s or "none": stretch(trained, s, args.factor) for s in (None, *SCALINGS)}
    if trained.rope_scaling:
        configs = {trained.rope_scaling: trained}
    results = {"model": name, "window": window, "length": length, "pages": len(pages), "bin": BIN, "methods": {}}
    print(f"loss on {len(pages)} held-out pages of {length:,} tokens, by position\n")
    print(f"{'method':<16}" + "".join(f"{f'{a:,}-{a + window - 1:,}':>13}" for a in range(0, length, window)))
    for method, config in {"window": trained, **configs}.items():
        model.config = config
        if method == "window":
            losses = torch.stack([window_losses(model, ids, window) for ids in pages])
        else:
            losses = torch.stack([full_losses(model, ids) for ids in pages])
        by_window = losses.view(len(pages), -1, window).mean(dim=(0, 2))
        results["methods"][method] = {
            "bins": [round(x, 4) for x in losses.view(len(pages), -1, BIN).mean(dim=(0, 2)).tolist()],
            "windows": [round(x, 4) for x in by_window.tolist()],
            "in_window": round(losses[:, :window].mean().item(), 4),
            "past_window": round(losses[:, window:].mean().item(), 4),
        }
        print(f"{method:<16}" + "".join(f"{x:>13.3f}" for x in by_window.tolist()), flush=True)

    os.makedirs(os.path.join(RESULTS, "long"), exist_ok=True)
    with open(os.path.join(RESULTS, "long", f"{name}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
