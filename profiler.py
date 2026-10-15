import argparse
import json
import os
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from bench import sync
from checkpoint import load_checkpoint
from model import pick_device

PROMPT = "Once upon a time"
TOKENS = 100


def profiled(device):
    # On a Mac's GPU, the profiler sees the CPU's side: handing each operation over to the GPU.
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    return profile(activities=activities)


def summary(prof, label):
    run = next(e for e in prof.events() if e.name == label)
    # The operations the run called itself, not the ones they call in turn, such as addmm inside linear.
    calls = [e for e in prof.events() if e.cpu_parent is not None and e.cpu_parent.id == run.id]
    ops = {}
    for e in calls:
        count, total = ops.get(e.name, (0, 0.0))
        ops[e.name] = (count + 1, total + e.cpu_time_total)
    rows = sorted(ops.items(), key=lambda kv: -kv[1][1])
    return {
        "ms": run.cpu_time_total / 1000,
        "calls": len(calls),
        "ops": [{"name": name, "calls": count, "share": total / run.cpu_time_total} for name, (count, total) in rows],
    }


@torch.no_grad()
def profile_pass(model, x):
    model(x)  # a first pass, to warm up
    with profiled(x.device) as prof:
        with record_function("pass"):
            model(x)
        sync(x.device)
    return prof


@torch.no_grad()
def profile_writing(model, prompt, tokens=TOKENS):
    model.generate(prompt, 10)  # a short run first, to warm up
    with profiled(prompt.device) as prof:
        with record_function("writing"):
            model.generate(prompt, tokens, temperature=0.8, top_k=50)
        sync(prompt.device)
    return prof


@torch.no_grad()
def ms_per_token(model, prompt, tokens=TOKENS, runs=5):
    times = []
    for _ in range(runs):
        sync(prompt.device)
        start = time.perf_counter()
        model.generate(prompt, tokens, temperature=0.8, top_k=50)
        sync(prompt.device)
        times.append((time.perf_counter() - start) * 1000 / tokens)
    return statistics.median(times)


def print_ops(s, rows=8):
    print(f"{'operation':<38}{'calls':>7}{'time':>8}")
    for op in s["ops"][:rows]:
        print(f"{op['name']:<38}{op['calls']:>7}{op['share']:>8.1%}")


def main():
    parser = argparse.ArgumentParser(description="Profile the model, operation by operation.")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--out", help="a folder to write the summaries and a trace to")
    args = parser.parse_args()

    torch.manual_seed(1337)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()

    # A full context window: each operation has enough work for its time to be its own.
    x = torch.randint(model.config.vocab_size, (1, model.config.block_size), device=args.device)
    prof = profile_pass(model, x)
    full = summary(prof, "pass")
    print(f"one pass over {x.size(1)} tokens: {full['ms']:.1f} ms, {full['calls']} operations\n")
    print_ops(full)

    # Writing, a token at a time with the cache: lots of small operations.
    prompt = torch.tensor([tokenizer.encode(PROMPT)], device=args.device)
    writing = summary(profile_writing(model, prompt), "writing")
    plain = ms_per_token(model, prompt)
    print(f"\nwriting {TOKENS} tokens: {writing['calls'] / TOKENS:.0f} operations per token,")
    print(f"{writing['ms'] / TOKENS:.2f} ms per token profiled, {plain:.2f} ms without the profiler\n")
    print_ops(writing, rows=5)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"profile-{args.device}.json"), "w") as f:
            json.dump({"pass": full, "writing": {**writing, "tokens": TOKENS, "ms_unprofiled": plain}}, f, indent=1)
        # A timeline of the pass, operation by operation, for Perfetto, at ui.perfetto.dev.
        prof.export_chrome_trace(os.path.join(args.out, f"trace-{args.device}.json"))


if __name__ == "__main__":
    main()
