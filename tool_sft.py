import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from model import pick_device
from tool_data import load
from tools import generate_with_tools

RUN_DIR = os.path.join(CHECKPOINTS, "tools")
BATCH_SIZE = 32
LR = 1e-4  # half of sft.py's: a model eight times the size, and a narrow task that shouldn't push out the rest
EVAL_EVERY = 250
DEMO = "Question: A ferry carries 128 people on each trip and makes 37 trips. How many people does it carry?\nAnswer:"


def encode(example, tokenizer, block_size):
    """An example's ids, and for each, whether the model learns to write it: the question and the calculator's
    results are given, the calls and the answers are learned."""
    ids, learn = [], []
    for text, learned in example["segments"]:
        piece = tokenizer.encode(text)
        ids += piece
        learn += [learned] * len(piece)
    return ids[: block_size + 1], learn[: block_size + 1]


def make_batch(examples, pad_id, device):
    longest = max(len(ids) for ids, _ in examples)
    x = np.full((len(examples), longest - 1), pad_id, dtype=np.int64)
    y = np.full((len(examples), longest - 1), -1, dtype=np.int64)
    for row, (ids, learn) in enumerate(examples):
        x[row, : len(ids) - 1] = ids[:-1]
        # Position t predicts token t + 1, which counts if it's one the model writes; the rest keep -1.
        targets = np.array(ids[1:])
        y[row, : len(ids) - 1] = np.where(learn[1:], targets, -1)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss(model, examples, pad_id):
    # All the held-out examples, in order, so each evaluation measures the same ones.
    model.eval()
    device = next(model.parameters()).device
    batches = range(0, len(examples), BATCH_SIZE)
    total = sum(model(*make_batch(examples[i : i + BATCH_SIZE], pad_id, device))[1].item() for i in batches)
    model.train()
    return total / len(batches)


def main(minutes):
    torch.manual_seed(1337)
    device = pick_device()
    model, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "pretrain-web", "model.pt"), device)
    block_size = model.config.block_size
    train = [encode(e, tokenizer, block_size) for e in load("train")]
    valid = [encode(e, tokenizer, block_size) for e in load("seen")]
    tokens = sum(sum(learn) for _, learn in train)
    print(f"{len(train):,} training examples, {tokens:,} tokens to learn, {len(valid):,} held out", flush=True)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95),
    )
    writer = SummaryWriter(os.path.join(RUNS, time.strftime("tools-%Y%m%d-%H%M%S")))
    demo = tokenizer.encode(DEMO)

    def evaluate(step):
        loss = estimate_loss(model, valid, tokenizer.pad_id)
        out, calls = generate_with_tools(model, tokenizer, demo, max_new_tokens=40)
        writer.add_scalar("loss/valid", loss, step)
        writer.add_text("sample", tokenizer.decode(out), step)
        writer.flush()
        print(f"step {step:>5} | valid {loss:.3f} | {tokenizer.decode(out).strip()}", flush=True)
        model.train()

    budget = minutes * 60
    start = time.time()
    step = 0
    evaluate(step)
    while time.time() - start < budget:
        picks = torch.randint(len(train), (BATCH_SIZE,)).tolist()
        x, y = make_batch([train[i] for i in picks], tokenizer.pad_id, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % 50 == 0:
            writer.add_scalar("loss/batch", loss.item(), step)
            print(f"step {step:>5} | loss {loss.item():.3f} | {(time.time() - start) / 60:.1f} min", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step)

    if step % EVAL_EVERY != 0:
        evaluate(step)
    writer.close()
    save_checkpoint(os.path.join(RUN_DIR, "model.pt"), model, tokenizer, step=step)
    print(f"done: {step:,} steps, {step * BATCH_SIZE:,} examples, {(time.time() - start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
