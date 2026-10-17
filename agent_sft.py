import os
import random
import shutil
import sys
import tempfile
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from agent import Agent, Sandbox, show
from agent_data import load
from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from model import pick_device
from tool_sft import encode, estimate_loss, make_batch

RUN_DIR = os.path.join(CHECKPOINTS, "agent")
BATCH_SIZE = 32  # at most, and fewer for long examples: a batch holds about BATCH_TOKENS
BATCH_TOKENS = 6144
LR = 1e-4  # tool_sft.py's
EVAL_EVERY = 250
DEMO = ("Add up the numbers in jan.txt and feb.txt.", {"jan.txt": "120, 45, 300", "feb.txt": "18, 7, 220, 64"})


def batches(examples, rng):
    """Random batches of examples of about the same length, so little of each batch is padding: shuffle, sort
    each stretch of 50 batches by length, cut it into batches of up to BATCH_TOKENS, and shuffle the batches.
    A batch of long examples holds fewer of them, so a batch's memory stays about the same."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    stretch = BATCH_SIZE * 50
    out = []
    for i in range(0, len(order), stretch):
        batch = []
        for j in sorted(order[i : i + stretch], key=lambda j: len(examples[j][0])):
            if batch and (len(batch) == BATCH_SIZE or (len(batch) + 1) * len(examples[j][0]) > BATCH_TOKENS):
                out.append(batch)
                batch = []
            batch.append(j)
        out.append(batch)
    rng.shuffle(out)
    return out


def main(minutes):
    torch.manual_seed(1337)
    rng = random.Random(1337)
    device = pick_device()
    # The model we pretrained on the web: one model, which learns all three roles.
    model, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "pretrain-web", "model.pt"), device)
    block_size = model.config.block_size
    train = [encode(e, tokenizer, block_size) for e in load("train")]
    valid = [encode(e, tokenizer, block_size) for e in load("agent")]
    print(f"{len(train):,} training examples, {sum(sum(learn) for _, learn in train):,} tokens to learn, "
          f"{len(valid):,} held out", flush=True)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95),
    )
    writer = SummaryWriter(os.path.join(RUNS, time.strftime("agent-%Y%m%d-%H%M%S")))

    def evaluate(step):
        loss = estimate_loss(model, valid, tokenizer.pad_id)
        folder = tempfile.mkdtemp()
        run = Agent(model, tokenizer, Sandbox(folder, DEMO[1])).run("agent", DEMO[0])
        shutil.rmtree(folder)
        writer.add_scalar("loss/valid", loss, step)
        writer.add_text("sample", show(run), step)
        writer.flush()
        print(f"step {step:>5} | valid {loss:.3f}\n" + show(run, "  "), flush=True)
        model.train()

    budget = minutes * 60
    start = time.time()
    step, seen, queue = 0, 0, []
    evaluate(step)
    while time.time() - start < budget:
        if not queue:
            queue = batches(train, rng)
        picks = queue.pop()
        x, y = make_batch([train[i] for i in picks], tokenizer.pad_id, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        seen += len(picks)
        if step % 50 == 0:
            # Batches come in many lengths, and PyTorch keeps memory for each shape it has seen: give it back.
            if device == "mps":
                torch.mps.empty_cache()
            writer.add_scalar("loss/batch", loss.item(), step)
            print(f"step {step:>5} | loss {loss.item():.3f} | {(time.time() - start) / 60:.1f} min", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step)

    if step % EVAL_EVERY != 0:
        evaluate(step)
    writer.close()
    save_checkpoint(os.path.join(RUN_DIR, "model.pt"), model, tokenizer, step=step)
    print(f"done: {step:,} steps, {seen:,} examples, {seen / len(train):.2f} passes, "
          f"{(time.time() - start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 30)
