import argparse
import json
import math
import os
import statistics
import time
from contextlib import ExitStack
from unittest import mock

import torch
from torch.nn import functional as F

import model as mvgpt
from bench import STORY, sync
from checkpoint import load_checkpoint
from model import GPT, GPTConfig, pick_device
from profiler import PROMPT, TOKENS, ms_per_token, profile_writing, summary


def attention_by_hand(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    # scaled_dot_product_attention, written out, for the arguments model.py passes it.
    T, S = q.size(-2), k.size(-2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if is_causal:
        attn_mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril()
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    return F.dropout(scores.softmax(dim=-1), dropout_p) @ v


# Each decision is a set of patches: model.py runs as it is, with a function swapped for another.
DECISIONS = {
    "as written": [],
    "GELU as one operation": [lambda: mock.patch.object(mvgpt, "gelu", F.gelu)],
    "attention by hand": [lambda: mock.patch.object(F, "scaled_dot_product_attention", attention_by_hand)],
}


@torch.no_grad()
def ms_per_pass(model, x, runs=10):
    model(x)  # a first pass, to warm up
    times = []
    for _ in range(runs):
        sync(x.device)
        start = time.perf_counter()
        model(x)
        sync(x.device)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


@torch.no_grad()
def measure(model, x, prompt, story):
    return {
        "pass_ms": ms_per_pass(model, x),
        "token_ms": ms_per_token(model, prompt),
        "ops_per_token": summary(profile_writing(model, prompt), "writing")["calls"] / TOKENS,
        "logits": model(story)[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Measure how decisions in the model's code change how fast it writes.")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--out", help="a folder to write the measurements to, as JSON")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    prompt = torch.tensor([tokenizer.encode(PROMPT)], device=args.device)
    story = torch.tensor([tokenizer.encode(STORY)], device=args.device)

    # A full context window, for a pass, and a short prompt, for writing a token at a time with the cache.
    x = torch.randint(model.config.vocab_size, (1, model.config.block_size), device=args.device)
    rows = []
    print(f"{'decision':<24}{'pass':>9}{'token':>9}{'operations':>12}{'difference':>12}")
    for name, patches in DECISIONS.items():
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch())
            m = measure(model, x, prompt, story)
        if not rows:
            reference = m["logits"]
        difference = (m["logits"] - reference).abs().max().item()
        rows.append({"decision": name, "pass_ms": m["pass_ms"], "token_ms": m["token_ms"],
                     "ops_per_token": m["ops_per_token"], "difference": difference})
        print(f"{name:<24}{m['pass_ms']:>7.2f}ms{m['token_ms']:>7.2f}ms{m['ops_per_token']:>12.0f}{difference:>12.1e}")

    # torch.compile turns the pass into fused kernels, the first time it runs, for one shape of input.
    compiled = torch.compile(model)
    compiled(x)
    pass_ms = ms_per_pass(compiled, x)
    difference = (compiled(story)[0] - reference).abs().max().item()
    rows.append({"decision": "torch.compile", "pass_ms": pass_ms, "difference": difference})
    print(f"{'torch.compile':<24}{pass_ms:>7.2f}ms{'':>9}{'':>12}{difference:>12.1e}")

    # The head scores each token of the vocabulary, so its size follows the vocabulary's.
    # An untrained model is as fast as a trained one, so there's no need to train one per vocabulary.
    print(f"\n{'vocabulary':<24}{'pass':>9}{'token':>9}{'head':>12}")
    vocab_rows = []
    for vocab_size in (5_000, 10_000, 20_000):
        torch.manual_seed(1337)
        untrained = GPT(GPTConfig(vocab_size=vocab_size)).to(args.device).eval()
        pass_ms, token_ms = ms_per_pass(untrained, x % vocab_size), ms_per_token(untrained, prompt)
        head = untrained.lm_head.weight.numel()
        vocab_rows.append({"vocab_size": vocab_size, "pass_ms": pass_ms, "token_ms": token_ms, "head_params": head})
        print(f"{vocab_size:<24,}{pass_ms:>7.2f}ms{token_ms:>7.2f}ms{head / 1e6:>11.2f}M")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"decisions-{args.device}.json"), "w") as f:
            json.dump({"decisions": rows, "vocabulary": vocab_rows}, f, indent=1)


if __name__ == "__main__":
    main()
