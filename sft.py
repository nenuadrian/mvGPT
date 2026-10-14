"""Supervised fine-tuning: the pretrained model learns to write the story an instruction asks for.

    python sft.py 10
"""
import re

import numpy as np
import torch

from tokenizer import EOT

BATCH_SIZE = 48

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
