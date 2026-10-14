"""Supervised fine-tuning: the pretrained model learns to write the story an instruction asks for.

    python sft.py 10
"""
import json
import os
import re
import sys
import textwrap
import time

import numpy as np
import torch

from checkpoint import CHECKPOINTS, load_checkpoint, save_checkpoint
from data import DATA_DIR
from model import pick_device
from sample import write
from tokenizer import EOT

RUN_DIR = os.path.join(CHECKPOINTS, "sft")
BATCH_SIZE = 48
LR = 2e-4
EVAL_EVERY = 250
DEMO = "Features: Dialogue\nWords: dog, happy, park\nSummary: A dog makes a new friend at the park.\nStory:"

# The fields of an instruction, in the order our prompts give them, before the story.
FIELDS = ["Features", "Words", "Random sentence", "Summary"]
FIELD = re.compile(r"^(Features|Words|Random sentence|Summary|Story):", re.MULTILINE)


def parse(doc):
    # The split alternates names and texts: ["", "Words", " dog, park\n", "Story", " Once ...", ...].
    parts = FIELD.split(doc)
    return {name: text.strip() for name, text in zip(parts[1::2], parts[2::2])}


def instruction(fields):
    lines = [f"{name}: {fields[name]}" for name in FIELDS if name in fields]
    return "\n".join(lines + ["Story:"])


def build_examples(path, tokenizer, block_size, limit=None):
    with open(path, encoding="utf-8") as f:
        docs = [d for d in f.read().split(EOT) if d.strip()][:limit]
    examples = []
    for doc in docs:
        fields = parse(doc)
        if not fields.get("Story"):
            continue
        prompt = tokenizer.encode(instruction(fields))
        story = tokenizer.encode(fields["Story"]) + [tokenizer.eot_id]
        ids = (prompt + story)[: block_size + 1]
        # An instruction that fills the window leaves nothing to learn from.
        if len(ids) > len(prompt):
            examples.append((ids, len(prompt)))
    return examples


def make_batch(examples, pad_id, device):
    longest = max(len(ids) for ids, _ in examples)
    x = np.full((len(examples), longest - 1), pad_id, dtype=np.int64)
    y = np.full((len(examples), longest - 1), -1, dtype=np.int64)
    for row, (ids, prompt_len) in enumerate(examples):
        x[row, : len(ids) - 1] = ids[:-1]
        # Only the story is learned: the last prompt position predicts its first token, and the
        # instruction and the padding keep their -1, which the loss skips.
        y[row, prompt_len - 1 : len(ids) - 1] = ids[prompt_len:]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def random_batch(examples, pad_id, device):
    picks = torch.randint(len(examples), (BATCH_SIZE,)).tolist()
    return make_batch([examples[i] for i in picks], pad_id, device)


@torch.no_grad()
def estimate_loss(model, examples, pad_id, batches=20):
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    for _ in range(batches):
        x, y = random_batch(examples, pad_id, device)
        total += model(x, y)[1].item()
    model.train()
    return total / batches


def main(minutes):
    torch.manual_seed(1337)
    device = pick_device()
    model, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "pretrain", "model.pt"), device)
    block_size = model.config.block_size
    train = build_examples(os.path.join(DATA_DIR, "instruct_train.txt"), tokenizer, block_size)
    valid = build_examples(os.path.join(DATA_DIR, "instruct_valid.txt"), tokenizer, block_size, limit=5000)
    print(f"{len(train):,} training examples, {len(valid):,} held out")

    # A lower learning rate than pretraining: we adjust what the model knows, not replace it.
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
        losses = {"train": estimate_loss(model, train, tokenizer.pad_id),
                  "valid": estimate_loss(model, valid, tokenizer.pad_id)}
        story = write(model, tokenizer, DEMO, max_new_tokens=120, with_prompt=False)
        log["evals"].append({"step": step, "minutes": round(elapsed / 60, 2), "train": round(losses["train"], 4),
                             "valid": round(losses["valid"], 4), "sample": story})
        print(f"step {step:>5} | train {losses['train']:.3f} | valid {losses['valid']:.3f}", flush=True)
        print(textwrap.indent(textwrap.fill(story, width=86), "  "), flush=True)
        os.makedirs(RUN_DIR, exist_ok=True)
        with open(os.path.join(RUN_DIR, "log.json"), "w") as f:
            json.dump(log, f, indent=1)

    budget = minutes * 60
    start = time.time()
    step = 0
    evaluate(step, 0)
    while time.time() - start < budget:
        x, y = random_batch(train, tokenizer.pad_id, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 50 == 0:
            log["steps"].append({"step": step, "loss": round(loss.item(), 4)})
            print(f"step {step:>5} | loss {loss.item():.3f} | {(time.time() - start) / 60:.1f} min", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step, time.time() - start)

    if log["evals"][-1]["step"] != step:
        evaluate(step, time.time() - start)
    save_checkpoint(os.path.join(RUN_DIR, "model.pt"), model, tokenizer, step=step)
    print(f"done: {step:,} steps, {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
