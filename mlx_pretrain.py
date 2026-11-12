import math
import os
import sys
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from checkpoint import CHECKPOINTS
from data import DATA_DIR, load_tokenizer
from mlx_model import GPT, to_torch
from pretrain import BATCH_SIZE, EVAL_EVERY, LR, MIN_LR, WARMUP_STEPS

# pretrain.py's run, in MLX: the model, batch and learning rate schedule, and AdamW, for a number of steps. Two
# differences: MLX's fused attention has no dropout, and its AdamW leaves out the bias correction by default.


def get_batch(data, block_size, batch_size):
    starts = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = np.stack([data[i : i + block_size] for i in starts]).astype(np.int32)
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in starts]).astype(np.int32)
    return mx.array(x), mx.array(y)


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def main(steps):
    np.random.seed(1337)
    mx.random.seed(1337)
    tokenizer = load_tokenizer()
    config = dict(vocab_size=len(tokenizer), block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1, bias=True)
    model = GPT(config)
    # MLX is lazy: the weights exist once something asks for them, so eval makes them now.
    mx.eval(model.parameters())
    splits = {s: np.memmap(os.path.join(DATA_DIR, f"{s}.bin"), dtype=np.uint16, mode="r") for s in ("train", "valid")}
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"{n_params:,} parameters on {mx.default_device()}, {len(splits['train']):,} training tokens")

    def learning_rate(step):
        # pretrain.py's schedule: a warmup over 100 steps, then a cosine down to a tenth, by the share of the steps.
        warm = LR * (step + 1) / WARMUP_STEPS
        cosine = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + mx.cos(math.pi * step / steps))
        return mx.where(step < WARMUP_STEPS, warm, cosine)

    # Weight decay on the matrices, and none on the biases and the norms' gains, as pretrain.py's param_groups does.
    optimizer = optim.MultiOptimizer(
        [optim.AdamW(learning_rate, betas=[0.9, 0.95], weight_decay=0.1),
         optim.AdamW(learning_rate, betas=[0.9, 0.95], weight_decay=0.0)],
        [lambda path, weight: weight.ndim >= 2])
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    # One training step, compiled: MLX traces it once and fuses what it can into fewer GPU kernels.
    @partial(mx.compile, inputs=state, outputs=state)
    def step_fn(x, y):
        loss, grads = loss_and_grad(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss

    def estimate_loss():
        model.eval()
        losses = {}
        for name, data in splits.items():
            losses[name] = sum(loss_fn(model, *get_batch(data, 256, BATCH_SIZE)).item() for _ in range(20)) / 20
        model.train()
        return losses

    model.train()
    start = last = time.time()
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0 or step == steps:
            losses = estimate_loss()
            print(f"step {step:>5} | train {losses['train']:.3f} | valid {losses['valid']:.3f}", flush=True)
            last = time.time()
        if step == steps:
            break
        loss = step_fn(*get_batch(splits["train"], 256, BATCH_SIZE))
        mx.eval(state)
        if (step + 1) % 50 == 0:
            now = time.time()
            print(f"step {step + 1:>5} | loss {loss.item():.3f} | {50 * BATCH_SIZE * 256 / (now - last):,.0f} tokens/s",
                  flush=True)
            last = now
    to_torch(model, tokenizer, os.path.join(CHECKPOINTS, "pretrain-mlx", "model.pt"))
    print(f"done: {steps:,} steps, {steps * BATCH_SIZE * 256:,} tokens, {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1250)
