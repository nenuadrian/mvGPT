import copy
import os
import sys
import textwrap
import time

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from data import DATA_DIR
from evaluate import load_instructions, words_used
from model import pick_device
from preferences import load_pairs
from sample import write
from sft import make_batch

RUN_DIR = os.path.join(CHECKPOINTS, "dpo")
BATCH_SIZE = 16  # pairs per step: 16 chosen stories and 16 rejected ones
LR = 1e-5
BETA = 0.1  # how far the policy may move from the reference to widen a margin
EVAL_EVERY = 200
DEMO = "Words: rain, cake, jump\nStory:"


def log_probs(model, x, y):
    # The log-probability of each sequence's story: the sum over its tokens, skipping the -1s.
    logits, _ = model(x)
    picked = F.log_softmax(logits, dim=-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * (y != -1)).sum(dim=-1)


def dpo_loss(policy, reference, pairs, pad_id, device):
    # Chosen stories in the first half of the batch, rejected ones in the second.
    examples = [(p["prompt"] + p["chosen"], len(p["prompt"])) for p in pairs]
    examples += [(p["prompt"] + p["rejected"], len(p["prompt"])) for p in pairs]
    x, y = make_batch(examples, pad_id, device)
    with torch.no_grad():
        ref = log_probs(reference, x, y)
    # How much more likely the policy makes each story than the reference does, in log-probability.
    gain = log_probs(policy, x, y) - ref
    margin = BETA * (gain[: len(pairs)] - gain[len(pairs) :])
    return -F.logsigmoid(margin).mean(), margin


@torch.no_grad()
def held_out(policy, reference, pairs, pad_id, device):
    margins = torch.cat([dpo_loss(policy, reference, pairs[i : i + 32], pad_id, device)[1]
                         for i in range(0, len(pairs), 32)])
    return {"loss": -F.logsigmoid(margins).mean().item(), "accuracy": (margins > 0).float().mean().item(),
            "margin": margins.mean().item()}


@torch.no_grad()
def words_rate(model, tokenizer, instructions):
    used = sum(words_used(write(model, tokenizer, prompt, with_prompt=False), words) for prompt, words in instructions)
    return used / sum(len(words) for _, words in instructions)


def main(minutes):
    torch.manual_seed(1337)
    device = pick_device()
    policy, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "sft", "model.pt"), device)
    # The reference is the fine-tuned model, frozen: DPO measures each change against it.
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    # Dropout stays off, so the two models' log-probabilities differ by training alone.
    policy.eval()
    pairs = load_pairs()
    train, valid = pairs["train"], pairs["valid"]
    instructions = load_instructions(os.path.join(DATA_DIR, "instruct_valid.txt"), 50)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    writer = SummaryWriter(os.path.join(RUNS, time.strftime("dpo-%Y%m%d-%H%M%S")))
    print(f"{len(train):,} training pairs, {len(valid)} held out", flush=True)

    def evaluate(step):
        m = held_out(policy, reference, valid, tokenizer.pad_id, device)
        m["words_used"] = words_rate(policy, tokenizer, instructions)
        story = write(policy, tokenizer, DEMO, max_new_tokens=120, with_prompt=False)
        for name, value in m.items():
            writer.add_scalar(f"held_out/{name}", value, step)
        writer.add_text("sample", story, step)
        writer.flush()
        print(f"step {step:>5} | held-out accuracy {m['accuracy']:.0%} | margin {m['margin']:+.2f} | "
              f"words used {m['words_used']:.1%}", flush=True)
        print(textwrap.indent(textwrap.fill(story, width=86), "  "), flush=True)
        save_checkpoint(os.path.join(RUN_DIR, f"step-{step}.pt"), policy, tokenizer, step=step)

    budget = minutes * 60
    start = time.time()
    step = 0
    evaluate(step)
    while time.time() - start < budget:
        batch = [train[i] for i in torch.randint(len(train), (BATCH_SIZE,)).tolist()]
        loss, margin = dpo_loss(policy, reference, batch, tokenizer.pad_id, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 20 == 0:
            accuracy = (margin > 0).float().mean().item()
            writer.add_scalar("loss/batch", loss.item(), step)
            print(f"step {step:>5} | loss {loss.item():.3f} | accuracy {accuracy:.0%} | "
                  f"{(time.time() - start) / 60:.1f} min", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step)

    if step % EVAL_EVERY != 0:
        evaluate(step)
    writer.close()
    save_checkpoint(os.path.join(RUN_DIR, "model.pt"), policy, tokenizer, step=step)
    print(f"done: {step:,} steps, {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
