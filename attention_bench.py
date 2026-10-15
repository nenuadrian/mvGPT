import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time

import torch
from torch.autograd import DeviceType
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from attention import attention_by_hand
from bench import sync


def pytorch(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


METHODS = {"by-hand": attention_by_hand, "pytorch": pytorch}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def inputs(args, T):
    torch.manual_seed(0)
    shape = (args.batch, args.heads, T, 64)
    return (torch.randn(shape, device=args.device, dtype=DTYPES[args.dtype]) for _ in range(3))


def milliseconds(fn, q, k, v, runs):
    fn(q, k, v)  # a first run, to warm up
    times = []
    for _ in range(runs):
        sync(q.device)
        start = time.perf_counter()
        fn(q, k, v)
        sync(q.device)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def peak_memory(name, T, args):
    # The most memory attention takes on top of q, k and v, in bytes.
    if args.device == "cuda":
        q, k, v = inputs(args, T)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        METHODS[name](q, k, v)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - before
    if args.device == "cpu":
        # On the CPU, from a process of its own: the most memory it ever held, less what it held before.
        flags = ["--batch", str(args.batch), "--heads", str(args.heads), "--dtype", args.dtype]
        out = subprocess.run([sys.executable, __file__, "--device", "cpu", "--peak", name, str(T)] + flags,
                             capture_output=True, text=True, check=True)
        return int(out.stdout)
    return None  # the Mac's GPU doesn't say how much memory it held at most


def max_rss():
    # Linux gives it in kilobytes, macOS in bytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def what_pytorch_runs(args):
    # Which of its attentions PyTorch picks here and, on an NVIDIA GPU, the kernels it launches, from its profiler.
    q, k, v = inputs(args, 1024)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if args.device == "cuda" else [])
    with profile(activities=activities) as prof:
        pytorch(q, k, v)
        sync(q.device)
    events = prof.key_averages()
    ops = [e.key for e in events if e.key.startswith("aten::_scaled_dot_product")]
    kernels = [e.key.removeprefix("void ").split("<")[0].split("(")[0] for e in events if e.device_type == DeviceType.CUDA]
    return ops + kernels


def flops(T, args):
    # Two products of a T x T table and the 64 channels, half of it masked away: 2 x 2 x T x T x 64 / 2, per head.
    return 2 * args.batch * args.heads * T * T * 64


def main():
    parser = argparse.ArgumentParser(description="Time causal attention, and measure the memory it takes, by length.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dtype", default="float32", choices=list(DTYPES))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", help="a JSON file to write the measurements to")
    parser.add_argument("--peak", nargs=2, metavar=("METHOD", "LENGTH"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.peak:  # in a process of its own, for peak_memory
        q, k, v = inputs(args, int(args.peak[1]))
        before = max_rss()
        METHODS[args.peak[0]](q, k, v)
        print(max_rss() - before)
        return

    device = torch.cuda.get_device_name() if args.device == "cuda" else args.device
    print(f"{args.batch} x {args.heads} heads x 64 channels, {args.dtype}, on {device}")
    print("PyTorch's attention here:", ", ".join(what_pytorch_runs(args)))
    rows = []
    for T in args.lengths:
        print(f"{T:,} tokens")
        for name in args.methods:
            try:
                q, k, v = inputs(args, T)
                ms = milliseconds(METHODS[name], q, k, v, args.runs)
                del q, k, v
                memory = peak_memory(name, T, args)
            except torch.OutOfMemoryError:
                print(f"  {name:<16}out of memory", flush=True)
                rows.append({"length": T, "method": name, "oom": True})
                torch.cuda.empty_cache()
                continue
            tflops = flops(T, args) / ms / 1e9
            mb = "" if memory is None else f"{memory / 2**20:>10,.0f} MB"
            speed = f"{tflops:>8.1f} TFLOP/s" if args.device == "cuda" else ""
            print(f"  {name:<16}{ms:>10,.2f} ms{mb}{speed}", flush=True)
            rows.append({"length": T, "method": name, "ms": ms, "bytes": memory, "tflops": tflops})
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"device": device, "batch": args.batch, "heads": args.heads, "dtype": args.dtype, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
