import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from long import FACTOR, full_losses, long_pages, long_spans
from model import SCALINGS, pick_device, stretch

ACCUM = 4  # pages per step, one at a time: two at once don't fit in a laptop's memory
LR = 5e-5
WARMUP = 10  # steps, then the learning rate stays where it is, as in the YaRN paper
EVAL_EVERY = 20
EVAL_PAGES = 8


def main(scaling, minutes):
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    device = pick_device()
    model, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "pretrain-web", "model.pt"), device)
    window = model.config.block_size
    model.config = stretch(model.config, scaling, FACTOR)
    length = model.config.block_size
    ids, starts, ends = long_spans(tokenizer, length, "train")
    print(f"{len(starts):,} training pages of more than {length:,} tokens, {(ends - starts).sum():,} tokens in all", flush=True)
    valid = long_pages(tokenizer, length, EVAL_PAGES, device)

    def page():
        # length + 1 tokens from a random place in a random long page.
        i = rng.integers(len(starts))
        s = rng.integers(starts[i], ends[i] - length)
        x = torch.from_numpy(ids[s : s + length + 1].astype(np.int64))[None].to(device)
        return x[:, :-1], x[:, 1:]

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    writer = SummaryWriter(os.path.join(RUNS, time.strftime(f"long-{scaling}-%Y%m%d-%H%M%S")))

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        losses = torch.stack([full_losses(model, x) for x in valid])
        inside, past = losses[:, :window].mean().item(), losses[:, window:].mean().item()
        writer.add_scalar("loss/in_window", inside, step)
        writer.add_scalar("loss/past_window", past, step)
        writer.flush()
        print(f"step {step:>4} | valid, in the window {inside:.3f} | past it {past:.3f}", flush=True)
        model.train()

    budget = minutes * 60
    start = time.time()
    step = 0
    evaluate(step)
    model.train()
    while time.time() - start < budget:
        # A short warmup, so the first steps' updates don't knock the model off what it learned.
        for group in optimizer.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        total = 0.0
        for _ in range(ACCUM):
            x, y = page()
            _, loss = model(x, y)
            # Divided by ACCUM, the gradients add up to the mean over the pages, as one batch would give.
            (loss / ACCUM).backward()
            total += loss.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        writer.add_scalar("loss/train", total, step)
        if step % 5 == 0:
            elapsed = time.time() - start
            print(f"step {step:>4} | loss {total:.3f} | {elapsed / 60:.1f} min | "
                  f"{step * ACCUM * length / elapsed:,.0f} tokens/s", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step)

    if step % EVAL_EVERY != 0:
        evaluate(step)
    writer.close()
    save_checkpoint(os.path.join(CHECKPOINTS, f"long-{scaling}", "model.pt"), model, tokenizer, step=step)
    print(f"done: {step:,} steps, {step * ACCUM:,} pages, {step * ACCUM * length:,} tokens, "
          f"{(time.time() - start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    scaling = sys.argv[1] if len(sys.argv) > 1 else "yarn"
    if scaling not in SCALINGS:
        sys.exit(f"the scaling is one of {', '.join(SCALINGS)}")
    main(scaling, float(sys.argv[2]) if len(sys.argv) > 2 else 15)
