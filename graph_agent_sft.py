import os
import random
import shutil
import sys
import tempfile
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from agent import Agent, Sandbox, show
from agent_sft import batches
from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from graph_agent_data import load
from model import pick_device
from tool_sft import LR, encode, estimate_loss, make_batch

RUN_DIR = os.path.join(CHECKPOINTS, "graph-agent")
EVAL_EVERY = 250
# Two weights from the graph and a subtraction: mvGPT, smollerGraph and the calculator in one answer, about two
# species training never names (250,000 and 90,000 grams in the graph).
DEMO = "How much heavier is the Okapi than the Baikal Seal?"


def main(minutes):
    torch.manual_seed(1337)
    rng = random.Random(1337)
    device = pick_device()
    # Episode 24's model: the agent, fine-tuned to write Cypher, which it has to learn to do as a role of its own.
    model, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "graph", "model.pt"), device)
    block_size = model.config.block_size
    train = [encode(e, tokenizer, block_size) for e in load("train")]
    valid = [encode(e, tokenizer, block_size) for e in load("seen")]
    print(f"{len(train):,} training examples, {sum(sum(learn) for _, learn in train):,} tokens to learn, "
          f"{len(valid):,} held out", flush=True)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95),
    )
    writer = SummaryWriter(os.path.join(RUNS, time.strftime("graph-agent-%Y%m%d-%H%M%S")))

    def evaluate(step):
        loss = estimate_loss(model, valid, tokenizer.pad_id)
        # The demo asks sg serve, as the agent will: it has to be running.
        folder = tempfile.mkdtemp()
        run = Agent(model, tokenizer, Sandbox(folder)).run("agent", DEMO)
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
