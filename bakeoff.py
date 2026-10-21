import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from checkpoint import RUNS
from data import DATA_DIR, load_tokenizer
from hellaswag import RESULTS
from model import GPT, GPTConfig, pick_device
from optimizers import AdamW

STEPS = 1000
WARMUP = 100
BATCH_SIZE = 48
EVAL_EVERY = 100
EVAL_BATCHES = 40  # held-out batches, one set for all the runs: 40 x 48 x 256 = 491,520 tokens
NAMES = ("adamw",)


def make_optimizers(name, model, lr):
    # Weight decay on the weight matrices, and not on the biases and the norms' gains, as in pretrain.py.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay}, {"params": rest, "weight_decay": 0.0}]
    if name == "adamw":
        return [AdamW(groups, lr=lr)]
    raise ValueError(f"no optimizer called {name}: one of {NAMES}")


def learning_rate(step, peak, steps):
    if step < WARMUP:
        return peak * (step + 1) / WARMUP
    # Cosine from the peak down to a tenth of it, at the last step.
    progress = (step - WARMUP) / (steps - WARMUP)
    return peak / 10 + 0.5 * (peak - peak / 10) * (1 + math.cos(math.pi * progress))


def batches(data, generator, block_size, device):
    # Random windows of the tokens, in the order the generator gives: a seed fixes the batches, whatever trains.
    while True:
        starts = torch.randint(len(data) - block_size - 1, (BATCH_SIZE,), generator=generator).tolist()
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts])
        yield x.to(device), y.to(device)


def sync(device):
    # The GPU runs behind Python: wait for it, so a clock reads the time the work took.
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def held_out_loss(model, valid):
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in valid) / len(valid)
    model.train()
    return loss


def main(name, lr, seed=0, steps=STEPS, device=None):
    device = device or pick_device()
    tokenizer = load_tokenizer(bpe=True)
    split = {part: np.memmap(os.path.join(DATA_DIR, f"{part}-bpe.bin"), dtype=np.uint16, mode="r")
             for part in ("train", "valid")}
    # The stories model, with rotary embeddings and the BPE tokenizer. No dropout, as 1,000 steps read about 60%
    # of the stories' tokens, and dropout is for a model that sees its data many times.
    config = GPTConfig(vocab_size=len(tokenizer), rope=True, dropout=0.0)
    torch.manual_seed(seed)
    model = GPT(config).to(device)
    optimizers = make_optimizers(name, model, lr)
    train = batches(split["train"], torch.Generator().manual_seed(seed), config.block_size, device)
    # The held-out batches come from a seed of their own, so all the runs are measured on one set of tokens.
    held_out = batches(split["valid"], torch.Generator().manual_seed(1234), config.block_size, device)
    valid = [next(held_out) for _ in range(EVAL_BATCHES)]

    run = f"{name}-{lr:g}-seed{seed}" + (f"-steps{steps}" if steps != STEPS else "")
    writer = SummaryWriter(os.path.join(RUNS, "bakeoff", run))
    print(f"{name} at a peak learning rate of {lr:g}, seed {seed}: {model.num_params():,} parameters on {device}")
    evals, trained, stepped = [], 0.0, 0.0

    def evaluate(step, lr_now, loss=None):
        evals.append({"step": step, "valid": held_out_loss(model, valid)})
        writer.add_scalar("loss/valid", evals[-1]["valid"], step)
        print(f"step {step:>4} | held-out {evals[-1]['valid']:.3f}" + (f" | loss {loss:.3f}" if loss else "")
              + f" | lr {lr_now:.1e} | {1000 * trained / max(step, 1):.0f} ms/step", flush=True)

    evaluate(0, learning_rate(0, lr, steps))
    for step in range(steps):
        lr_now = learning_rate(step, lr, steps)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = lr_now
        x, y = next(train)
        sync(device)
        start = time.time()
        loss = model(x, y)[1]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sync(device)
        between = time.time()
        for optimizer in optimizers:
            optimizer.step()
        sync(device)
        # The whole step's time, and the optimizer's share of it.
        trained += time.time() - start
        stepped += time.time() - between
        writer.add_scalar("loss/train", loss.item(), step + 1)
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == steps:
            evaluate(step + 1, lr_now, loss.item())

    # What each optimizer keeps between steps, in numbers per weight.
    kept = sum(t.numel() for o in optimizers for s in o.state.values() for t in s.values() if torch.is_tensor(t))
    result = {
        "name": name, "lr": lr, "seed": seed, "steps": steps, "tokens": steps * BATCH_SIZE * config.block_size,
        "valid": evals[-1]["valid"], "evals": evals, "ms_per_step": 1000 * trained / steps,
        "ms_optimizer": 1000 * stepped / steps, "params": model.num_params(), "state": kept,
    }
    os.makedirs(os.path.join(RESULTS, "bakeoff"), exist_ok=True)
    with open(os.path.join(RESULTS, "bakeoff", f"{run}.json"), "w") as f:
        json.dump(result, f, indent=1)
    writer.close()
    print(f"done: held-out {result['valid']:.3f} after {steps:,} steps, {result['tokens']:,} tokens, "
          f"{result['ms_per_step']:.0f} ms a step, {result['ms_optimizer']:.0f} of them in the optimizer, "
          f"{kept / result['params']:.2f} numbers kept per weight")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the stories model with an optimizer, for a fixed number of steps.")
    parser.add_argument("name", choices=NAMES)
    parser.add_argument("lrs", type=float, nargs="+", help="peak learning rates, one run each")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    for lr in args.lrs:
        main(args.name, lr, args.seed, args.steps, args.device)
