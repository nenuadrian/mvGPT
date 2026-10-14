"""Pretraining: the model learns to predict the next token of TinyStories, for a set number of minutes.

    python pretrain.py 10
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

from checkpoint import CHECKPOINTS, save_checkpoint
from data import DATA_DIR, load_tokenizer
from model import GPT, GPTConfig, pick_device
from sample import write

RUN_DIR = os.path.join(CHECKPOINTS, "pretrain")
BATCH_SIZE = 48
LR = 6e-4
MIN_LR = LR / 10
WARMUP_STEPS = 100
EVAL_EVERY = 250


def get_batch(data, block_size, batch_size, device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts])
    # The targets are the inputs shifted by one: position t learns to predict token t + 1.
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, batches=20):
    model.eval()
    device = next(model.parameters()).device
    losses = {}
    for name, data in splits.items():
        total = 0.0
        for _ in range(batches):
            x, y = get_batch(data, model.config.block_size, BATCH_SIZE, device)
            total += model(x, y)[1].item()
        losses[name] = total / batches
    model.train()
    return losses


def learning_rate(step, progress):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    # Cosine from LR down to MIN_LR, finishing as the time runs out.
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def main(minutes):
    torch.manual_seed(1337)
    device = pick_device()
    tokenizer = load_tokenizer()
    config = GPTConfig(vocab_size=len(tokenizer))
    model = GPT(config).to(device)
    splits = {
        split: np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")
        for split in ("train", "valid")
    }
    print(f"{model.num_params():,} parameters on {device}, {len(splits['train']):,} training tokens")

    # Weight decay pulls the weight matrices towards zero, but not the biases and LayerNorm gains.
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(0.9, 0.95),
    )

    log = {"steps": [], "evals": []}

    def evaluate(step, elapsed):
        losses = estimate_loss(model, splits)
        story = write(model, tokenizer, "Once upon a time", max_new_tokens=40)
        log["evals"].append({"step": step, "minutes": round(elapsed / 60, 2), "train": round(losses["train"], 4),
                             "valid": round(losses["valid"], 4), "sample": story})
        print(f"step {step:>5} | train {losses['train']:.3f} | valid {losses['valid']:.3f} | {story}", flush=True)
        # A checkpoint at each evaluation, to compare the model as it learns.
        save_checkpoint(os.path.join(RUN_DIR, f"step-{step}.pt"), model, tokenizer, step=step)
        with open(os.path.join(RUN_DIR, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    budget = minutes * 60
    start = last = time.time()
    step = 0
    evaluate(step, 0)
    while (elapsed := time.time() - start) < budget:
        lr = learning_rate(step, elapsed / budget)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(splits["train"], config.block_size, BATCH_SIZE, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # A rare bad batch can't throw the weights off: the gradient's length is capped at 1.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 50 == 0:
            now = time.time()
            speed = 50 * BATCH_SIZE * config.block_size / (now - last)
            last = now
            log["steps"].append({"step": step, "loss": round(loss.item(), 4), "lr": lr, "tokens_per_s": round(speed)})
            print(f"step {step:>5} | loss {loss.item():.3f} | lr {lr:.1e} | {speed:,.0f} tokens/s", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step, time.time() - start)
            last = time.time()

    if log["evals"][-1]["step"] != step:
        evaluate(step, time.time() - start)
    save_checkpoint(os.path.join(RUN_DIR, "model.pt"), model, tokenizer, step=step)
    print(f"done: {step:,} steps, {step * BATCH_SIZE * config.block_size:,} tokens, "
          f"{(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
