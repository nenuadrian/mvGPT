import argparse
import os

import numpy as np
import torch

from checkpoint import load_checkpoint
from data import DATA_DIR
from model import pick_device, rotary

# A sparse autoencoder on the residual stream after one block: each token's channels written as a sum of a few
# directions, out of many thousands, the features, which we can then read one at a time.


def stream(model, idx, layer):
    """The residual stream after the first `layer` blocks, without running the rest of the model."""
    c = model.config
    x = model.embed_tokens(idx)
    angles = None
    if c.rope:
        angles = rotary(0, idx.size(1), c.n_embd // c.n_head, idx.device)
    else:
        x = x + model.embed_positions(torch.arange(idx.size(1), device=idx.device))
    for block in model.blocks[:layer]:
        x = block(x, None, angles)
    return x


def get_windows(data, window, count, device):
    starts = np.random.randint(0, len(data) - window, count)
    return torch.from_numpy(np.stack([data[i : i + window].astype(np.int64) for i in starts])).to(device)


@torch.no_grad()
def norms(args):
    # How long the stream's vectors are, position by position, averaged over held-out windows.
    device = pick_device()
    model, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    data = np.memmap(os.path.join(DATA_DIR, "valid-web.bin"), dtype=np.uint16, mode="r")
    np.random.seed(0)
    lengths = stream(model, get_windows(data, 1024, 16, device), args.layer).norm(dim=-1).mean(0).tolist()
    for i in range(6):
        print(f"position {i}: {lengths[i]:6.1f}")
    print(f"positions 6 to 1023: {np.median(lengths[6:]):6.1f} on average (median)")


def main():
    parser = argparse.ArgumentParser(description="A sparse autoencoder on a model's residual stream.")
    sub = parser.add_subparsers(dest="command", required=True)
    n = sub.add_parser("norms")
    n.add_argument("checkpoint")
    n.add_argument("--layer", type=int, default=6)
    args = parser.parse_args()
    {"norms": norms}[args.command](args)


if __name__ == "__main__":
    main()
