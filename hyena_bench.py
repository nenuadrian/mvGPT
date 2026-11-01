import argparse
import json
import math
import os
import statistics
import time
from contextlib import nullcontext

import torch

from bench import sync
from hyena import Hyena
from model import CausalSelfAttention, GPTConfig, rotary


def layer_flops(kind, T, C, heads):
    # Multiply-adds count 2. Both layers project each token in (3C x C) and out (C x C): 8 T C^2 for a sequence.
    projections = 8 * T * C * C
    if kind == "attention":
        # Scores q k^T and the weighted sum of v, 2 T^2 C each, half of them past the diagonal and skipped.
        return projections + 2 * T * T * C
    # Hyena: for each channel, an FFT of the gated values and an inverse one, about 5 n log2 n with n = 2T padded,
    # a complex product per frequency, the two gates, the skip and the short filter, about 30 T C between them.
    n = 2 * T
    return projections + C * (5 * n * math.log2(n) + 30 * T)


def make_layer(kind, T, C, heads, device):
    config = GPTConfig(block_size=T, n_embd=C, n_head=heads, dropout=0.0, rope=True)
    layer = CausalSelfAttention(config, 0) if kind == "attention" else Hyena(config, 0)
    return layer.to(device)


def time_layer(kind, T, args):
    torch.manual_seed(0)
    layer = make_layer(kind, T, args.embd, args.heads, args.device)
    batch = max(1, args.tokens // T)
    x = torch.randn(batch, T, args.embd, device=args.device, requires_grad=True)
    grad = torch.randn_like(x)
    angles = rotary(0, T, args.embd // args.heads, args.device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()

    def run():
        with amp:
            y = layer(x, angles=angles) if kind == "attention" else layer(x)
        y.backward(grad)
        x.grad = None
        layer.zero_grad(set_to_none=True)

    run()  # a first run to warm up
    times = []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.runs):
        sync(x.device)
        start = time.perf_counter()
        run()
        sync(x.device)
        times.append((time.perf_counter() - start) * 1000)
    memory = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
    return batch, statistics.median(times), memory


def main():
    parser = argparse.ArgumentParser(description="One attention layer against one Hyena layer, forward and backward, "
                                     "the same number of tokens at each length.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096, 16384, 65536])
    parser.add_argument("--tokens", type=int, default=65536, help="tokens a batch: 64 windows of 1,024, 1 of 65,536")
    parser.add_argument("--embd", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", help="a JSON file to write the measurements to")
    args = parser.parse_args()

    device = torch.cuda.get_device_name() if args.device == "cuda" else args.device
    print(f"{args.tokens:,} tokens a batch, {args.embd} channels, forward and backward, on {device}")
    print(f"{'tokens':>8} {'windows':>8} {'attention':>12} {'Hyena':>12} {'faster':>8}   GFLOP: attention, Hyena")
    rows = []
    for T in args.lengths:
        row = {"length": T, "batch": max(1, args.tokens // T)}
        for kind in ("attention", "hyena"):
            try:
                _, ms, memory = time_layer(kind, T, args)
                row.update({f"{kind}_ms": ms, f"{kind}_bytes": memory})
            except torch.OutOfMemoryError:
                row[f"{kind}_ms"] = None
                torch.cuda.empty_cache()
            # Per batch, for the forward pass: the backward does about twice as much again.
            row[f"{kind}_gflop"] = layer_flops(kind, T, args.embd, args.heads) * row["batch"] / 1e9
        a, h = row["attention_ms"], row["hyena_ms"]
        times = "".join(f"{f'{x:,.1f} ms' if x else 'out of memory':>12}" for x in (a, h))
        faster = f"{a / h:>7.1f}x" if a and h else ""
        print(f"{T:>8,} {row['batch']:>8} {times} {faster}   {row['attention_gflop']:,.0f}, {row['hyena_gflop']:,.0f}",
              flush=True)
        rows.append(row)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"device": device, "tokens": args.tokens, "embd": args.embd, "heads": args.heads, "rows": rows},
                      f, indent=1)


if __name__ == "__main__":
    main()
