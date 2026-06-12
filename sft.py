"""Stage 2: supervised fine-tuning (instruction tuning) on TinyStories-Instruct.

Same next-token objective as pretraining, different data: (instruction ->
story) pairs. Loss counts only the story tokens - the prompt is masked to -1 -
so the model learns to write a story conditioned on the instruction, not to
reproduce instructions. EOT terminates each example to teach stopping.

Loads ckpt_base.pt (from pretrain.py), produces ckpt_sft.pt.

Run:  python sft.py [minutes]      (default 5)
"""

import os
import sys
import time

import numpy as np
import torch

from base_model import (CKPT_BASE, CKPT_SFT, DATA_DIR, EOT, device,
                        load_checkpoint, sample_story, save_checkpoint)
from pretrain import ensure_data

MARKER = "Story:"

# fixed hand-written prompt (not from val): does it follow Words/Features and stop at EOT?
DEMO_PROMPT = ("Features: Dialogue\nWords: dog, happy, park\n"
               "Summary: A dog makes a new friend at the park.\nStory:")


# Parse the instruct corpus -> list of (token ids, prompt length). prompt_len
# tells make_batch which positions to mask.
def build_examples(path, tok, block_size, eot, limit=None):
    with open(path, encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split(EOT) if d.strip()][:limit]

    examples = []
    for d in docs:
        i = d.find(MARKER)
        if i == -1:
            continue
        prompt = tok.encode(d[: i + len(MARKER)])
        comp = tok.encode(d[i + len(MARKER):].strip())
        if not comp:
            continue
        ids = (prompt + comp + [eot])[: block_size + 1]
        if len(ids) > len(prompt):  # truncation must leave some completion
            examples.append((ids, len(prompt)))
    return examples


# Build one padded batch. Inputs pad with PAD, targets with -1, so padding is
# ignored by the loss. Targets at positions 0..plen-2 are also -1: those predict
# prompt tokens. The last prompt position predicts the first story token, so
# masking stops one short of plen. Right-padding is safe under causal attention.
def make_batch(examples, idxs, pad_id, device):
    maxlen = max(len(examples[i][0]) for i in idxs)
    X = np.full((len(idxs), maxlen - 1), pad_id, dtype=np.int64)
    Y = np.full((len(idxs), maxlen - 1), -1, dtype=np.int64)
    for r, i in enumerate(idxs):
        ids, plen = examples[i]
        X[r, : len(ids) - 1] = ids[:-1]
        Y[r, plen - 1 : len(ids) - 1] = ids[plen:]
    return torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)


# Completion-masked loss on a held-out example set. Counts only story tokens,
# so it's NOT comparable to pretraining loss - track it over time: val rising
# while train falls = memorizing the SFT set.
@torch.no_grad()
def estimate_sft_loss(model, examples, batch_size, pad_id, iters=20):
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    losses = []
    for _ in range(iters):
        idxs = torch.randint(len(examples), (batch_size,)).tolist()
        _, loss = model(*make_batch(examples, idxs, pad_id, dev))
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / iters


def main(minutes=5.0):
    torch.manual_seed(42)
    np.random.seed(42)

    ensure_data()
    model, tok, _ = load_checkpoint(CKPT_BASE)
    cfg = model.cfg

    max_iters = 100_000
    batch_size = 48
    lr = 2e-4  # lower than pretraining: nudge behavior, don't overwrite the base
    weight_decay = 0.1
    betas = (0.9, 0.95)
    eval_interval = 250

    examples = build_examples(os.path.join(DATA_DIR, "instruct_train.txt"),
                              tok, cfg.block_size, tok.eot_id, limit=40_000)
    val_examples = build_examples(os.path.join(DATA_DIR, "instruct_valid.txt"),
                                  tok, cfg.block_size, tok.eot_id, limit=5000)
    print(f"SFT examples: {len(examples):,}  (held-out val {len(val_examples):,})", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ], lr=lr, betas=betas)

    model.train()
    t0 = time.time()
    deadline = t0 + minutes * 60
    it = 0

    while it < max_iters and time.time() < deadline:
        idxs = torch.randint(len(examples), (batch_size,)).tolist()
        x, y = make_batch(examples, idxs, tok.pad_id, device)
        _, loss = model(x, y)  # prompt/pad targets are -1, so loss covers story tokens only
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        it += 1

        if it % 50 == 0:
            print(f"iter {it:5d} | loss {loss.item():.3f} | {(time.time() - t0) / 60:.1f} min", flush=True)

        if it % eval_interval == 0:
            val_loss = estimate_sft_loss(model, val_examples, batch_size, tok.pad_id)
            print(f"  >> eval iter {it}: val_loss {val_loss:.3f}", flush=True)
            story = sample_story(model, tok, DEMO_PROMPT, n=120, eos_id=tok.eot_id)
            print("  --- sample story ---\n  " + story.replace("\n", "\n  "), flush=True)
            save_checkpoint(CKPT_SFT, model, tok, iters=it)

    save_checkpoint(CKPT_SFT, model, tok, iters=it)
    print(f"\nDONE  iters={it}  time={(time.time() - t0) / 60:.1f} min")
    print(f"saved {CKPT_SFT}")

    for i in range(3):
        print(f"\n--- sample {i} ---\n"
              f"{sample_story(model, tok, DEMO_PROMPT, n=160, eos_id=tok.eot_id)}")


if __name__ == "__main__":
    main(minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 5.0)
