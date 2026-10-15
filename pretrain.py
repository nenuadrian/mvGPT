import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, save_checkpoint
from data import DATA_DIR, load_tokenizer
from diffusion import diffusion_loss, from_autoregressive, mask_id
from diffusion import write as write_by_denoising
from model import GPT, GPTConfig, pick_device
from sample import write

BATCH_SIZE = 48
LR = 6e-4
MIN_LR = LR / 10
WARMUP_STEPS = 100
EVAL_EVERY = 250


def get_batch(data, block_size, batch_size, device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts])
    # The targets are the inputs shifted by one: position t learns to predict token t + 1.
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


def next_token_loss(model, x, y):
    return model(x, y)[1]


@torch.no_grad()
def estimate_loss(model, splits, loss_of=next_token_loss, batches=20):
    model.eval()
    device = next(model.parameters()).device
    losses = {}
    for name, data in splits.items():
        total = 0.0
        for _ in range(batches):
            x, y = get_batch(data, model.config.block_size, BATCH_SIZE, device)
            total += loss_of(model, x, y).item()
        losses[name] = total / batches
    model.train()
    return losses


def learning_rate(step, progress):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    # Cosine from LR down to MIN_LR, finishing as the time runs out.
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def main(minutes, rope=False, bpe=False, diffusion=False, init=None):
    # Each kind of model gets a run folder of its own: pretrain, pretrain-rope, pretrain-rope-bpe, ...
    run_dir = os.path.join(CHECKPOINTS, "pretrain" + ("-rope" if rope else "") + ("-bpe" if bpe else "")
                           + ("-diffusion" if diffusion else "") + ("-converted" if init else ""))
    torch.manual_seed(1337)
    device = pick_device()
    tokenizer = load_tokenizer(bpe)
    # A diffusion model attends both ways, and has one more token, [MASK].
    config = GPTConfig(vocab_size=len(tokenizer) + 1 if diffusion else len(tokenizer), rope=rope, causal=not diffusion)
    model = GPT(config).to(device)
    if init:
        from_autoregressive(model, init)

    def loss_of(model, x, y):
        # Diffusion hides some of x's tokens and learns to fill them in; next-token prediction learns y.
        return diffusion_loss(model, x, mask_id(tokenizer)) if diffusion else next_token_loss(model, x, y)

    splits = {
        split: np.memmap(os.path.join(DATA_DIR, f"{split}{'-bpe' if bpe else ''}.bin"), dtype=np.uint16, mode="r")
        for split in ("train", "valid")
    }
    print(f"{model.num_params():,} parameters on {device}, {len(splits['train']):,} training tokens")

    # Weight decay pulls the weight matrices towards zero, but not the biases and LayerNorm gains.
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(0.9, 0.95),
    )

    # The run's numbers go to TensorBoard, in a folder named after the run and when it started.
    writer = SummaryWriter(os.path.join(RUNS, time.strftime(f"{os.path.basename(run_dir)}-%Y%m%d-%H%M%S")))

    def evaluate(step):
        losses = estimate_loss(model, splits, loss_of)
        if diffusion:
            story = write_by_denoising(model, tokenizer, "Once upon a time", tokens=44)
        else:
            story = write(model, tokenizer, "Once upon a time", max_new_tokens=40)
        writer.add_scalar("loss/train", losses["train"], step)
        writer.add_scalar("loss/valid", losses["valid"], step)
        writer.add_text("sample", story, step)
        # Written to disk now, so TensorBoard shows the evaluation while the run goes on.
        writer.flush()
        print(f"step {step:>5} | train {losses['train']:.3f} | valid {losses['valid']:.3f} | {story}", flush=True)
        # A checkpoint at each evaluation, to compare the model as it learns.
        save_checkpoint(os.path.join(run_dir, f"step-{step}.pt"), model, tokenizer, step=step)

    budget = minutes * 60
    start = last = time.time()
    step = 0
    evaluate(step)
    while (elapsed := time.time() - start) < budget:
        lr = learning_rate(step, elapsed / budget)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(splits["train"], config.block_size, BATCH_SIZE, device)
        loss = loss_of(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # A rare bad batch can't throw the weights off: the gradient's length is capped at 1.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % 50 == 0:
            now = time.time()
            speed = 50 * BATCH_SIZE * config.block_size / (now - last)
            last = now
            writer.add_scalar("loss/batch", loss.item(), step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("tokens_per_s", speed, step)
            print(f"step {step:>5} | loss {loss.item():.3f} | lr {lr:.1e} | {speed:,.0f} tokens/s", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step)
            last = time.time()

    if step % EVAL_EVERY != 0:
        evaluate(step)
    writer.close()
    save_checkpoint(os.path.join(run_dir, "model.pt"), model, tokenizer, step=step)
    print(f"done: {step:,} steps, {step * BATCH_SIZE * config.block_size:,} tokens, "
          f"{(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    # --init PATH starts a diffusion model from a next-token model's weights.
    init = sys.argv[sys.argv.index("--init") + 1] if "--init" in sys.argv else None
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10, rope="--rope" in sys.argv, bpe="--bpe" in sys.argv,
         diffusion="--diffusion" in sys.argv, init=init)
