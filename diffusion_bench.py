import argparse
import json
import math
import os
import statistics
import time

import torch
from torch.nn import functional as F

from bench import sync
from checkpoint import load_checkpoint
from data import DATA_DIR
from diffusion import canvas_for, denoise, mask_id
from model import pick_device
from tokenizer import EOT

PROMPT = "Once upon a time"


def by_denoising(model, tokenizer, steps):
    canvas = canvas_for(model, tokenizer, [tokenizer.eot_id] + tokenizer.encode(PROMPT))
    return denoise(model, canvas, mask_id(tokenizer), steps, temperature=0.8, top_k=50)[0].tolist()


def by_next_token(model, tokenizer):
    # A token at a time, with the KV cache, until end of text or the end of the window.
    device = next(model.parameters()).device
    ids = torch.tensor([[tokenizer.eot_id] + tokenizer.encode(PROMPT)], device=device)
    out = model.generate(ids, model.config.block_size - ids.size(1), temperature=0.8, top_k=50, eos_id=tokenizer.eot_id)
    return out[0].tolist()


@torch.no_grad()
def judged(judge, tokenizer, ids):
    # The judge's loss on the story after the prompt, end of text included: how likely a bigger model
    # finds what was written. Lower is more like the stories it learned from.
    start = 1 + len(tokenizer.encode(PROMPT))
    end = ids.index(tokenizer.eot_id, 1) + 1 if tokenizer.eot_id in ids[1:] else len(ids)
    x = torch.tensor([ids[:end]], device=next(judge.parameters()).device)
    logits, _ = judge(x[:, :-1])
    nats = F.cross_entropy(logits[0, start - 1 :], x[0, start:], reduction="sum").item()
    return nats, end - start, tokenizer.decode(ids[start:end])


def distinct(story):
    # The share of the story's runs of three words that it doesn't repeat: 100% for no repetition.
    words = story.split()
    triples = list(zip(words, words[1:], words[2:]))
    return len(set(triples)) / max(len(triples), 1)


def measure(write, judge, tokenizer, stories, device):
    write()  # once to warm up, untimed
    times, nats, tokens, unique = [], 0.0, 0, []
    for _ in range(stories):
        sync(device)
        start = time.perf_counter()
        ids = write()
        sync(device)
        times.append(time.perf_counter() - start)
        n, length, story = judged(judge, tokenizer, ids)
        nats, tokens = nats + n, tokens + length
        unique.append(distinct(story))
    return {"ms": statistics.median(times) * 1000, "tokens": tokens / stories,
            "judge_perplexity": math.exp(nats / tokens), "distinct": statistics.mean(unique)}


def held_out(judge, tokenizer, stories):
    # Stories from the held-out set that start with the prompt, scored the same way: what a real story gets.
    with open(os.path.join(DATA_DIR, "valid.txt"), encoding="utf-8") as f:
        texts = [t.strip() for t in f.read(2_000_000).split(EOT) if t.strip().startswith(PROMPT)][:stories]
    nats, tokens, unique = 0.0, 0, []
    for text in texts:
        n, length, story = judged(judge, tokenizer, [tokenizer.eot_id] + tokenizer.encode(text) + [tokenizer.eot_id])
        nats, tokens = nats + n, tokens + length
        unique.append(distinct(story))
    return {"tokens": tokens / len(texts), "judge_perplexity": math.exp(nats / tokens), "distinct": statistics.mean(unique)}


def main():
    parser = argparse.ArgumentParser(description="Write stories by denoising, with more or fewer steps, and a token at a time: time them, and score them with a bigger model.")
    parser.add_argument("diffusion", help="a diffusion model's checkpoint")
    parser.add_argument("--against", required=True, help="a next-token model's checkpoint, to compare with")
    parser.add_argument("--judge", required=True, help="a bigger next-token model, which scores the stories")
    parser.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--stories", type=int, default=16)
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--out", help="a file to write the measurements to, as JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    diffusion, tokenizer, _ = load_checkpoint(args.diffusion, device)
    against, _, _ = load_checkpoint(args.against, device)
    judge, _, _ = load_checkpoint(args.judge, device)
    for model in (diffusion, against, judge):
        model.eval()

    print(f"{args.stories} stories each, on {args.device}, scored by {os.path.basename(os.path.dirname(args.judge))}\n")
    print(f"{'':<22}{'steps':>6}{'ms/story':>10}{'tokens':>8}{'judge ppl':>11}{'distinct':>10}")
    rows = [{"writer": "held-out stories", **held_out(judge, tokenizer, args.stories)}]
    torch.manual_seed(0)
    rows.append({"writer": "next token", "steps": None,
                 **measure(lambda: by_next_token(against, tokenizer), judge, tokenizer, args.stories, device)})
    for steps in args.steps:
        torch.manual_seed(0)
        rows.append({"writer": "diffusion", "steps": steps,
                     **measure(lambda: by_denoising(diffusion, tokenizer, steps), judge, tokenizer, args.stories, device)})
    for row in rows:
        steps = "" if row.get("steps") is None else row["steps"]
        ms = f"{row['ms']:,.0f}" if "ms" in row else ""
        print(f"{row['writer']:<22}{steps:>6}{ms:>10}{row['tokens']:>8.0f}{row['judge_perplexity']:>11.2f}"
              f"{row['distinct']:>10.1%}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
