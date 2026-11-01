"""Multi-query associative recall, MQAR (Arora et al., 2023, "Zoology"): pairs of a key and a value, then the keys
again, here and there, each to be answered with its value. Attention can find a key by matching it against the
tokens before; a convolution weighs tokens by how far back they are, so it has to carry the pairs in its channels.

    k1 v1 k2 v2 ... kN vN  0 0 k2 [v2] 0 k1 [v1] 0 0 ...
"""

import argparse
import json
import math
import os
import time

import torch
from torch.nn import functional as F

from hellaswag import RESULTS
from model import GPT, GPTConfig, pick_device

VOCAB = 8192  # 0 is the filler, keys are 1 to 4,095 and values 4,096 to 8,191


def examples(batch, length, pairs, generator, device):
    # Each example draws its own keys and values: nothing to remember across examples, only within one. Sorting
    # random numbers gives each row its own random order, so the first `pairs` of 1 to 4,095 are its keys.
    keys = torch.rand(batch, VOCAB // 2 - 1, generator=generator).argsort(dim=1)[:, :pairs] + 1
    values = torch.randint(VOCAB // 2, VOCAB, (batch, pairs), generator=generator)
    x = torch.zeros(batch, length, dtype=torch.long)
    x[:, 0 : 2 * pairs : 2], x[:, 1 : 2 * pairs : 2] = keys, values
    # Every key asked once, in another order, at random places after the pairs: in a slot of two tokens, the key
    # and then its value.
    slots = torch.rand(batch, (length - 2 * pairs) // 2, generator=generator).argsort(dim=1)[:, :pairs]
    where = 2 * pairs + 2 * slots.sort(dim=1).values
    order = torch.rand(batch, pairs, generator=generator).argsort(dim=1)
    x.scatter_(1, where, keys.gather(1, order))
    x.scatter_(1, where + 1, values.gather(1, order))
    # Only a query's next token is to learn and to score: the value of the key just read. The rest is -1.
    y = torch.full_like(x, -1)
    y.scatter_(1, where, values.gather(1, order))
    return x.to(device), y.to(device)


def model_for(mixer, width, length):
    # Two blocks, as in the Zoology paper: attention in both, Hyena in both, or Hyena then attention.
    config = GPTConfig(vocab_size=VOCAB, block_size=length, n_layer=2, n_head=max(1, width // 32), n_embd=width,
                       dropout=0.0, rope=True, mixer="attention" if mixer == "attention" else "hyena",
                       attention_layers=(1,) if mixer == "hybrid" else ())
    return GPT(config)


@torch.no_grad()
def accuracy(model, length, pairs, device, batches=8):
    g = torch.Generator().manual_seed(1234)  # the same held-out examples for every model
    right = total = 0
    for _ in range(batches):
        x, y = examples(64, length, pairs, g, "cpu")
        logits, _ = model(x.to(device))
        asked = y != -1
        right += (logits.argmax(dim=-1).cpu()[asked] == y[asked]).sum().item()
        total += asked.sum().item()
    return right / total


def train(mixer, width, length, mix, steps, lr, device, seed=0):
    torch.manual_seed(seed)
    model = model_for(mixer, width, length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.98))
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        # A linear warm-up over the first tenth, then a cosine down to zero.
        warm = steps // 10
        scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
        for group in optimizer.param_groups:
            group["lr"] = lr * scale
        # A curriculum: with five numbers of pairs, the first fifth of the steps on the fewest, then each fifth adds
        # the next, and each batch draws from those allowed so far. Drawn from the whole mix from the start, the
        # hard batches drown the easy ones, and nothing learns the lookup.
        allowed = mix[: 1 + step * len(mix) // steps]
        pairs = allowed[torch.randint(len(allowed), (1,), generator=g).item()]
        x, y = examples(64, length, pairs, g, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1), ignore_index=-1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train two-block models on multi-query associative recall, and "
                                     "measure how many keys they answer, by mixer, width and number of pairs.")
    parser.add_argument("--mixers", nargs="+", default=["attention", "hyena", "hybrid"])
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--pairs", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                        help="each model trains on these numbers of pairs, the fewest first, and is scored on each")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3], help="the best of these, for each model")
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--out", default=os.path.join(RESULTS, "recall.json"))
    args = parser.parse_args()

    print(f"{args.length} tokens, {args.steps:,} steps of 64 examples, keys answered right:")
    print(f"{'mixer':<11}{'width':>6}" + "".join(f"{f'{n} pairs':>11}" for n in args.pairs))
    rows = []
    for mixer in args.mixers:
        for width in args.widths:
            start = time.time()
            # For each learning rate, a model trained on the mix and scored at each number of pairs; the best on average.
            scores = []
            for lr in args.lrs:
                model = train(mixer, width, args.length, args.pairs, args.steps, lr, args.device)
                scores.append([accuracy(model, args.length, pairs, args.device) for pairs in args.pairs])
            best = max(scores, key=sum)
            for pairs, right in zip(args.pairs, best):
                rows.append({"mixer": mixer, "width": width, "pairs": pairs, "accuracy": right})
            rows[-1]["seconds"] = round(time.time() - start, 1)
            print(f"{mixer:<11}{width:>6}" + "".join(f"{r:>11.1%}" for r in best), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"length": args.length, "steps": args.steps, "lrs": args.lrs, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
