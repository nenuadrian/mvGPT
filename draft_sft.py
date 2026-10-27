import math
import os
import random
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from agent import prompt
from agent_sft import batches
from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from graph_agent_data import load
from model import GPT, GPTConfig, pick_device
from speculative import Draft, passes
from tool_sft import encode, estimate_loss, make_batch

RUN_DIR = os.path.join(CHECKPOINTS, "draft")
# A small GPT for speculative decoding: four layers of 384 channels, about a fifth of the agent, with its tokenizer,
# so each of its tokens is one the agent can check.
CONFIG = dict(n_layer=4, n_head=6, n_embd=384)
LR = 1e-3  # a small model from scratch takes a larger step than a fine-tune
WARMUP = 100
EVAL_EVERY = 500
DEMO = "How much does the King Penguin weigh?"


@torch.no_grad()
def agreement(draft, valid, pad_id):
    """The share of the held-out tokens the agent learned to write that the draft's likeliest token gets right: about
    the share of its guesses the agent would keep, one at a time."""
    draft.eval()
    device = next(draft.parameters()).device
    right = total = 0
    for i in range(0, len(valid), 32):
        x, y = make_batch(valid[i : i + 32], pad_id, device)
        logits, _ = draft(x)
        learned = y != -1
        right += (logits.argmax(dim=-1) == y)[learned].sum().item()
        total += learned.sum().item()
    draft.train()
    return right / total


def main(minutes):
    torch.manual_seed(1337)
    rng = random.Random(1337)
    device = pick_device()
    # Episode 25's agent, for its tokenizer and its block size, and to check the draft's guesses in the demo.
    agent, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "graph-agent", "model.pt"), device)
    agent.eval()
    config = GPTConfig(vocab_size=len(tokenizer), block_size=agent.config.block_size, rope=True, **CONFIG)
    model = GPT(config).to(device)
    print(f"the draft: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
          f"the agent: {sum(p.numel() for p in agent.parameters()) / 1e6:.1f}M", flush=True)
    # Episode 25's training set: the agent's own tasks and its subagents', which the agent was trained to write.
    train = [encode(e, tokenizer, config.block_size) for e in load("train")]
    valid = [encode(e, tokenizer, config.block_size) for e in load("seen")]
    print(f"{len(train):,} training examples, {sum(sum(learn) for _, learn in train):,} tokens to learn, "
          f"{len(valid):,} held out", flush=True)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95),
    )
    writer = SummaryWriter(os.path.join(RUNS, time.strftime("draft-%Y%m%d-%H%M%S")))

    def evaluate(step):
        loss, agree = estimate_loss(model, valid, tokenizer.pad_id), agreement(model, valid, tokenizer.pad_id)
        # The graph role's query for the demo, a pass at a time, with the draft guessing four tokens.
        ids = tokenizer.encode(prompt("graph", DEMO))
        out, count = [], 0
        for wrote, _ in passes(agent, ids, Draft(model.eval(), k=4)):
            out, count = out + wrote, count + 1
            if tokenizer.decode(out).endswith(">>") or len(out) > 60:
                break
        model.train()
        writer.add_scalar("loss/valid", loss, step)
        writer.add_scalar("agreement", agree, step)
        writer.flush()
        print(f"step {step:>5} | valid {loss:.3f} | agrees {agree:.1%} | the demo's query: {len(out)} tokens in "
              f"{count} passes", flush=True)

    budget = minutes * 60
    start = time.time()
    step, seen, queue = 0, 0, []
    evaluate(step)
    while time.time() - start < budget:
        # Warm up, then a cosine down to zero as the time runs out, as the run is measured in minutes, not steps.
        done = (time.time() - start) / budget
        lr = LR * min(1, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * done))
        for group in optimizer.param_groups:
            group["lr"] = lr
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
            if device == "mps":
                torch.mps.empty_cache()
            writer.add_scalar("loss/batch", loss.item(), step)
            print(f"step {step:>5} | loss {loss.item():.3f} | lr {lr:.1e} | {(time.time() - start) / 60:.1f} min",
                  flush=True)
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
