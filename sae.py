import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint
from data import DATA_DIR
from model import pick_device, rotary

# A sparse autoencoder on the residual stream after one block: each token's channels written as a sum of a few
# directions, out of many thousands, the features, which we can then read one at a time.


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, n_features, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        # The point the features are measured from: the encoder sees the stream minus it, and the decoder adds it back.
        self.bias = nn.Parameter(torch.zeros(d_model))
        # How much the stream is scaled by on the way in, so that its tokens are about the length of sqrt(d_model).
        self.register_buffer("scale", torch.ones(()))
        with torch.no_grad():
            # Each feature's direction starts random, with a length of 1, and the encoder as the decoder's
            # transpose, so that a feature first fires on the stream along its own direction.
            d = torch.randn(d_model, n_features)
            self.decoder.weight.copy_(d / d.norm(dim=0, keepdim=True))
            self.encoder.weight.copy_(self.decoder.weight.t())
            self.encoder.bias.zero_()

    def encode(self, x):
        pre = self.encoder(x - self.bias)
        # Each token keeps its k largest, through a ReLU, and the other features are zero.
        top = torch.topk(pre, self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, F.relu(top.values))

    def decode(self, features):
        return self.decoder(features) + self.bias

    def forward(self, x):
        features = self.encode(x)
        return self.decode(features), features

    @torch.no_grad()
    def normalize_decoder(self):
        # Each feature's direction kept at a length of 1, so that how strongly it's there is its activation alone.
        self.decoder.weight /= self.decoder.weight.norm(dim=0, keepdim=True)


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


def train(args):
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = pick_device()
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    data = np.memmap(os.path.join(DATA_DIR, "train-web.bin"), dtype=np.uint16, mode="r")
    d_model = model.config.n_embd
    sae = SparseAutoencoder(d_model, d_model * args.expansion, args.k).to(device)
    windows = args.batch // args.window

    @torch.no_grad()
    def batch():
        # Each window's first token is left out: the model makes its stream many times longer than the others' (see
        # norms), and it would take over the scale and the loss.
        x = stream(model, get_windows(data, args.window, windows, device), args.layer)
        return x[:, 1:].reshape(-1, d_model)

    # The scale that makes the stream's tokens about sqrt(d_model) long, from a first batch.
    sae.scale.fill_((d_model ** 0.5 / batch().norm(dim=-1).mean()).item())
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    run_dir = os.path.join(CHECKPOINTS, "sae")
    writer = SummaryWriter(os.path.join(RUNS, time.strftime("sae-%Y%m%d-%H%M%S")))
    # How many tokens since each feature last fired: one that hasn't in two million tokens counts as dead.
    since = torch.zeros(sae.encoder.out_features, device=device)
    print(f"{sae.encoder.out_features:,} features, {args.k} per token, on the stream after block {args.layer}")

    start = time.time()
    step = 0
    while time.time() - start < args.minutes * 60:
        x = batch() * sae.scale
        recon, features = sae(x)
        # The reconstruction's error as a share of the stream's variance: 0 is perfect, 1 is no better than the mean.
        unexplained = (recon - x).pow(2).sum(-1).mean() / (x - x.mean(0)).pow(2).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        unexplained.backward()
        optimizer.step()
        sae.normalize_decoder()
        since += len(x)
        since[(features > 0).any(0)] = 0
        step += 1
        if step % 100 == 0:
            dead = (since > 2_000_000).float().mean().item()
            writer.add_scalar("sae/unexplained", unexplained.item(), step)
            writer.add_scalar("sae/dead", dead, step)
            print(f"step {step:>5} | unexplained {unexplained.item():.3f} | dead {dead:.1%} | "
                  f"{step * args.batch / (time.time() - start):,.0f} tokens/s", flush=True)
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"sae": sae.state_dict(), "checkpoint": args.checkpoint, "layer": args.layer, "k": args.k,
                "n_features": sae.encoder.out_features}, os.path.join(run_dir, "sae.pt"))
    print(f"done: {step:,} steps, {step * args.batch:,} tokens, {(time.time() - start) / 60:.1f} minutes")


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
    t = sub.add_parser("train")
    t.add_argument("checkpoint")
    t.add_argument("--layer", type=int, default=6, help="the stream after this many blocks")
    t.add_argument("--expansion", type=int, default=16, help="features per channel of the stream")
    t.add_argument("--k", type=int, default=32, help="features per token")
    t.add_argument("--minutes", type=float, default=15)
    t.add_argument("--batch", type=int, default=4096, help="tokens a step")
    t.add_argument("--window", type=int, default=1024)
    t.add_argument("--lr", type=float, default=2e-4)
    n = sub.add_parser("norms")
    n.add_argument("checkpoint")
    n.add_argument("--layer", type=int, default=6)
    args = parser.parse_args()
    {"train": train, "norms": norms}[args.command](args)


if __name__ == "__main__":
    main()
