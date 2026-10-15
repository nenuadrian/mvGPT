import argparse
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, save_checkpoint
from data import DATA_DIR, load_tokenizer
from model import GPT, GPTConfig
from pretrain import EVAL_EVERY, LR, estimate_loss, get_batch, learning_rate
from sample import write


def setup():
    # torchrun starts a process per GPU, and tells each its rank, its GPU on this machine, and how many there are.
    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    else:
        device = "cpu"  # gloo, for trying it out on a machine without NVIDIA GPUs
        dist.init_process_group("gloo")
    return rank, world, device


def main():
    parser = argparse.ArgumentParser(description="Pretrain on several GPUs at once, with DistributedDataParallel.")
    parser.add_argument("minutes", type=float)
    # GPT-2 small's shape by default, with a smaller vocabulary: about 91 million parameters.
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--embd", type=int, default=768)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument("--batch", type=int, default=48, help="sequences per GPU per step")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--full", action="store_true", help="all of TinyStories: python data.py --full")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--name", default="pretrain-ddp")
    args = parser.parse_args()

    rank, world, device = setup()
    main_rank = rank == 0
    # Each rank draws different batches; with one seed, they'd all train on the same data.
    torch.manual_seed(1337 + rank)
    tokenizer = load_tokenizer(bpe=True)
    config = GPTConfig(vocab_size=len(tokenizer), block_size=args.block, n_layer=args.layers, n_head=args.heads,
                       n_embd=args.embd, dropout=args.dropout, rope=True)
    raw = GPT(config).to(device)
    # DDP gives all ranks rank 0's starting weights, and averages the gradients across them after each backward.
    model = DDP(torch.compile(raw) if args.compile else raw,
                device_ids=[int(device.split(":")[1])] if device.startswith("cuda") else None)
    train = "train-full-bpe.bin" if args.full else "train-bpe.bin"
    # The held-out stories are the ones the laptop's models were measured on.
    splits = {"train": np.memmap(os.path.join(DATA_DIR, train), dtype=np.uint16, mode="r"),
              "valid": np.memmap(os.path.join(DATA_DIR, "valid-bpe.bin"), dtype=np.uint16, mode="r")}
    if main_rank:
        gpu = torch.cuda.get_device_name() if device.startswith("cuda") else "CPU"
        print(f"{raw.num_params():,} parameters, {world} x {gpu}, {len(splits['train']):,} training tokens", flush=True)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95),
    )
    run_dir = os.path.join(CHECKPOINTS, args.name)
    writer = SummaryWriter(os.path.join(RUNS, time.strftime(f"{args.name}-%Y%m%d-%H%M%S"))) if main_rank else None
    # On NVIDIA GPUs, bfloat16 for the matrix multiplications: half the memory traffic, a lot more throughput.
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = True

    def evaluate(step):
        if main_rank:
            with amp:
                losses = estimate_loss(raw, splits)
                story = write(raw, tokenizer, "Once upon a time", max_new_tokens=40)
            writer.add_scalar("loss/train", losses["train"], step)
            writer.add_scalar("loss/valid", losses["valid"], step)
            writer.add_text("sample", story, step)
            writer.flush()
            print(f"step {step:>5} | train {losses['train']:.3f} | valid {losses['valid']:.3f} | {story}", flush=True)
            # Saved at each evaluation: a rented machine can go away mid-run.
            save_checkpoint(os.path.join(run_dir, "model.pt"), raw, tokenizer, step=step, gpus=world)
        # The other ranks wait here while rank 0 evaluates, so they all set off again together.
        dist.barrier(device_ids=[torch.cuda.current_device()] if device.startswith("cuda") else None)

    budget = args.minutes * 60
    start = last = time.time()
    step = 0
    evaluate(step)
    elapsed = torch.zeros(1, device=device)
    while True:
        # Rank 0's clock decides, for all of them: the same learning rate, and the same last step.
        elapsed[0] = time.time() - start
        dist.broadcast(elapsed, src=0)
        if elapsed.item() >= budget:
            break
        lr = learning_rate(step, elapsed.item() / budget)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(splits["train"], config.block_size, args.batch, device)
        with amp:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 50 == 0 and main_rank:
            now = time.time()
            # Tokens per second for all the GPUs together: each did a batch per step.
            speed = 50 * world * args.batch * config.block_size / (now - last)
            last = now
            writer.add_scalar("loss/batch", loss.item(), step)
            writer.add_scalar("tokens_per_s", speed, step)
            print(f"step {step:>5} | loss {loss.item():.3f} | lr {lr:.1e} | {speed:,.0f} tokens/s", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step)
            last = time.time()

    if step % EVAL_EVERY != 0:
        evaluate(step)
    if main_rank:
        writer.close()
        tokens = step * world * args.batch * config.block_size
        memory = f", {torch.cuda.max_memory_allocated() / 2**30:.1f} GB per GPU" if device.startswith("cuda") else ""
        print(f"done: {step:,} steps, {tokens:,} tokens, {(time.time() - start) / 60:.1f} minutes, "
              f"{tokens / (time.time() - start):,.0f} tokens/s{memory}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
