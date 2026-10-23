import os
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from graph import generate_with_graph
from graph_data import load
from model import pick_device
from tool_sft import BATCH_SIZE, LR, encode, estimate_loss, make_batch

RUN_DIR = os.path.join(CHECKPOINTS, "graph")
EVAL_EVERY = 250
# A species training never names, asked the way training asks: the model has to copy the name into its query.
DEMO = "Question: What family is the Red fox in?\nAnswer:"


def main(minutes):
    torch.manual_seed(1337)
    device = pick_device()
    # Episode 21's agent: the web model, already fine-tuned to write <<...>> calls, so it knows the shape of one.
    model, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "agent", "model.pt"), device)
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
    writer = SummaryWriter(os.path.join(RUNS, time.strftime("graph-%Y%m%d-%H%M%S")))
    demo = tokenizer.encode(DEMO)

    def evaluate(step):
        loss = estimate_loss(model, valid, tokenizer.pad_id)
        # The demo asks sg serve, as the model will: it has to be running.
        out, calls = generate_with_graph(model, tokenizer, demo, max_new_tokens=96)
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
    print(f"done: {step:,} steps, {step * BATCH_SIZE:,} examples, {step * BATCH_SIZE / len(train):.2f} passes, "
          f"{(time.time() - start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 20)
