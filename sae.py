import argparse
import os
import textwrap
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint
from data import DATA_DIR
from model import pick_device, rotary
from sample import write

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


def load_sae(path, device):
    saved = torch.load(path, map_location=device)
    model, tokenizer, _ = load_checkpoint(saved["checkpoint"], device)
    model.eval()
    sae = SparseAutoencoder(model.config.n_embd, saved["n_features"], saved["k"]).to(device)
    sae.load_state_dict(saved["sae"])
    return sae, model, tokenizer, saved["layer"]


class Splice:
    """While it's open, the stream after `layer` blocks goes through change(x) before the rest of the model sees it."""

    def __init__(self, model, layer, change):
        self.block, self.change = model.blocks[layer - 1], change

    def __enter__(self):
        # A forward hook that returns a value replaces the block's output with it.
        self.hook = self.block.register_forward_hook(lambda module, inputs, out: self.change(out))

    def __exit__(self, *exc):
        self.hook.remove()


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


@torch.no_grad()
def evaluate(args):
    device = pick_device()
    sae, model, _, layer = load_sae(args.sae, device)
    data = np.memmap(os.path.join(DATA_DIR, "valid-web.bin"), dtype=np.uint16, mode="r")
    np.random.seed(0)
    windows = get_windows(data, 1025, args.windows, device)
    x, y = windows[:, :-1], windows[:, 1:]

    def loss():
        total = 0.0
        for i in range(0, len(x), 4):
            logits, _ = model(x[i : i + 4])
            total += F.cross_entropy(logits.flatten(0, 1), y[i : i + 4].flatten(), reduction="sum").item()
        return total / y.numel()

    clean = loss()
    # The stream replaced by the autoencoder's reconstruction of it, scaled back to the model's units, past each
    # window's first token, which the autoencoder never saw.
    with Splice(model, layer, lambda s: torch.cat([s[:, :1], sae(s[:, 1:] * sae.scale)[0] / sae.scale], dim=1)):
        spliced = loss()
    # And by the stream's average token, which keeps none of what each token carries: the floor to compare with.
    mean = stream(model, x[:8], layer)[:, 1:].mean(dim=(0, 1))
    with Splice(model, layer, lambda s: torch.cat([s[:, :1], mean.expand_as(s[:, 1:])], dim=1)):
        ablated = loss()
    print(f"next-token loss on {y.numel():,} held-out tokens")
    print(f"  the model as it is:                 {clean:.3f}")
    print(f"  through the sparse autoencoder:     {spliced:.3f}")
    print(f"  with the average token instead:     {ablated:.3f}")
    print(f"  loss recovered by the autoencoder:  {(ablated - spliced) / (ablated - clean):.1%}")


def context(tokenizer, ids, at, before=12, after=3):
    # The text around token `at`, with the token itself in bold yellow.
    left = tokenizer.decode(ids[max(0, at - before) : at])
    token = tokenizer.decode(ids[at : at + 1])
    right = tokenizer.decode(ids[at + 1 : at + 1 + after])
    return (left + "\x1b[1;33m" + token + "\x1b[0m" + right).replace("\n", " ")


@torch.no_grad()
def fire_rates(sae, model, layer, device, tokens=50_000):
    # The share of held-out tokens each feature fires on, past each window's first token.
    data = np.memmap(os.path.join(DATA_DIR, "valid-web.bin"), dtype=np.uint16, mode="r")[:tokens]
    windows = torch.from_numpy(data[: len(data) // 1024 * 1024].astype(np.int64)).view(-1, 1024)
    fired = torch.zeros(sae.encoder.out_features, device=device)
    for i in range(0, len(windows), 8):
        fired += (sae.encode(stream(model, windows[i : i + 8].to(device), layer)[:, 1:] * sae.scale) > 0).sum(dim=(0, 1))
    return fired / (len(windows) * 1023)


@torch.no_grad()
def features_of(args):
    # Each token of a text, with the features that fire most strongly on it.
    device = pick_device()
    sae, model, tokenizer, layer = load_sae(args.sae, device)
    # A feature that fires on most tokens says little about any one of them, so those are left out of the list.
    dense = fire_rates(sae, model, layer, device) > 0.5
    print("left out, as they fire on over half of all tokens: " + ", ".join(map(str, dense.nonzero().flatten().tolist())))
    ids = tokenizer.encode(args.text)
    features = sae.encode(stream(model, torch.tensor([ids], device=device), layer)[0] * sae.scale)
    features[:, dense] = 0
    # The first token, as in training, is left out.
    for i, row in enumerate(features[1:], start=1):
        top = row.topk(args.top)
        found = ", ".join(f"{f} ({v:.1f})" for v, f in zip(top.values.tolist(), top.indices.tolist()) if v > 0)
        print(f"{tokenizer.decode([ids[i]])!r:>16}  {found}")


@torch.no_grad()
def show(args):
    # For each feature asked for: how often it fires on held-out text, the contexts where it fires most strongly, and
    # the tokens its direction pushes the model towards.
    device = pick_device()
    sae, model, tokenizer, layer = load_sae(args.sae, device)
    data = np.memmap(os.path.join(DATA_DIR, "valid-web.bin"), dtype=np.uint16, mode="r")[: args.tokens]
    wanted = torch.tensor(args.features, device=device)
    best = [[] for _ in args.features]
    fired = torch.zeros(len(args.features), device=device)
    for start in range(0, len(data) - 1024, 1024 * 8):
        windows = torch.from_numpy(np.stack([data[i : i + 1024].astype(np.int64)
                                             for i in range(start, min(start + 1024 * 8, len(data) - 1024), 1024)]))
        acts = sae.encode(stream(model, windows.to(device), layer)[:, 1:] * sae.scale)[..., wanted]
        fired += (acts > 0).sum(dim=(0, 1))
        for j in range(len(args.features)):
            top = acts[..., j].flatten().topk(args.examples)
            # Back from an index into the windows without their first token, to a position in the data.
            at = [start + (i // 1023) * 1024 + i % 1023 + 1 for i in top.indices.tolist()]
            best[j] += [(v, a) for v, a in zip(top.values.tolist(), at) if v > 0]
    ids = data.tolist()
    # A direction's effect on the next token, read straight through the head: which tokens it raises the most.
    directions = sae.decoder.weight[:, wanted].t() / sae.scale
    raised = (directions @ model.lm_head.weight.t()).topk(8).indices
    for j, f in enumerate(args.features):
        print(f"\nfeature {f}: fires on {fired[j].item() / len(data):.2%} of tokens, raises: "
              + ", ".join(repr(tokenizer.decode([t])) for t in raised[j].tolist()))
        for v, at in sorted(best[j], reverse=True)[: args.examples]:
            print(f"  {v:5.1f}  {context(tokenizer, ids, at)}")


@torch.no_grad()
def steer(args):
    device = pick_device()
    sae, model, tokenizer, layer = load_sae(args.sae, device)
    # The feature's direction, in the stream's own units, times how strongly we push.
    push = args.strength * sae.decoder.weight[:, args.feature] / sae.scale
    for strength, label in ((0, "as it is"), (args.strength, f"feature {args.feature} at {args.strength:g}")):
        torch.manual_seed(args.seed)
        with Splice(model, layer, lambda s: s + push * (strength > 0)):
            text = write(model, tokenizer, args.prompt, args.tokens)
        print(f"{label}:\n" + textwrap.fill(" ".join(text.split()), width=90, initial_indent="  ",
                                            subsequent_indent="  ") + "\n")


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
    e = sub.add_parser("evaluate")
    e.add_argument("sae")
    e.add_argument("--windows", type=int, default=64)
    f = sub.add_parser("features")
    f.add_argument("sae")
    f.add_argument("text")
    f.add_argument("--top", type=int, default=3)
    sh = sub.add_parser("show")
    sh.add_argument("sae")
    sh.add_argument("features", type=int, nargs="+")
    sh.add_argument("--tokens", type=int, default=500_000)
    sh.add_argument("--examples", type=int, default=6)
    st = sub.add_parser("steer")
    st.add_argument("sae")
    st.add_argument("feature", type=int)
    st.add_argument("prompt")
    st.add_argument("--strength", type=float, default=10)
    st.add_argument("--tokens", type=int, default=60)
    st.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"train": train, "norms": norms, "evaluate": evaluate, "features": features_of, "show": show, "steer": steer}[args.command](args)


if __name__ == "__main__":
    main()
