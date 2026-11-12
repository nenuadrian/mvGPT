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
from model import MODERN, GPT, GPTConfig, pick_device
from sample import write

BATCH_SIZE = 48
LR = 6e-4
MIN_LR = LR / 10
WARMUP_STEPS = 100
EVAL_EVERY = 250
BALANCE_WEIGHT = 0.01


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
    # Past windows of 512, fewer windows a batch and more batches, so the logits fit next to what training holds:
    # as many tokens in all, 24,576 a batch.
    batch = min(BATCH_SIZE, BATCH_SIZE * 512 // model.config.block_size)
    n = batches * BATCH_SIZE // batch
    for name, data in splits.items():
        total = 0.0
        for _ in range(n):
            x, y = get_batch(data, model.config.block_size, batch, device)
            total += loss_of(model, x, y).item()
        losses[name] = total / n
    model.train()
    return losses


def param_groups(model):
    # Weight decay pulls the weight matrices towards zero, but not the biases and the norms' gains, nor the small
    # network that draws Hyena's filters (hyena.py), which its authors leave out too.
    decay, rest = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and ".filter." not in name else rest).append(p)
    return [{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}]


def learning_rate(step, progress):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    # Cosine from LR down to MIN_LR, finishing as the time runs out.
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def main(minutes, rope=False, bpe=False, diffusion=False, init=None, modern=False, kv_heads=None, hyena=False,
         experts=0, top_k=2, balance="loss", steps=None):
    # Each kind of model gets a run folder of its own: pretrain, pretrain-rope, pretrain-rope-bpe, ...
    moe = f"-moe{experts}" + (f"-top{top_k}" if top_k != 2 else "") + (f"-{balance}" if balance != "loss" else "")
    run_dir = os.path.join(CHECKPOINTS, "pretrain" + ("-rope" if rope else "") + ("-bpe" if bpe else "")
                           + ("-modern" if modern else "") + (f"-kv{kv_heads}" if kv_heads else "")
                           + ("-hyena" if hyena else "") + ("-diffusion" if diffusion else "")
                           + ("-converted" if init else "") + (moe if experts else ""))
    torch.manual_seed(1337)
    device = pick_device()
    tokenizer = load_tokenizer(bpe)
    # A diffusion model attends both ways, and has one more token, [MASK].
    config = GPTConfig(vocab_size=len(tokenizer) + 1 if diffusion else len(tokenizer), rope=rope, causal=not diffusion,
                       n_kv_head=kv_heads, mixer="hyena" if hyena else "attention", n_expert=experts,
                       expert_top_k=top_k, **(MODERN if modern else {}))
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

    optimizer = torch.optim.AdamW(param_groups(model), lr=LR, betas=(0.9, 0.95))

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

    def progress():
        # How far through the run we are, from 0 to 1: by the clock, or with steps, by the steps taken, so that runs
        # of different speeds see as many tokens.
        return step / steps if steps else (time.time() - start) / budget

    while progress() < 1:
        lr = learning_rate(step, progress())
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(splits["train"], config.block_size, BATCH_SIZE, device)
        loss = loss_of(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        if experts and balance == "loss":
            # The experts' balancing loss, averaged over the blocks, a hundredth of it on top of the loss we learn
            # from, as in the Switch Transformer paper.
            (loss + BALANCE_WEIGHT * sum(b.mlp.aux_loss for b in model.blocks) / config.n_layer).backward()
        else:
            loss.backward()
        # A rare bad batch can't throw the weights off: the gradient's length is capped at 1.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if experts and balance == "bias":
            for block in model.blocks:
                block.mlp.update_bias()
        step += 1

        if step % 50 == 0:
            now = time.time()
            speed = 50 * BATCH_SIZE * config.block_size / (now - last)
            last = now
            writer.add_scalar("loss/batch", loss.item(), step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("tokens_per_s", speed, step)
            line = f"step {step:>5} | loss {loss.item():.3f} | lr {lr:.1e} | {speed:,.0f} tokens/s"
            if experts:
                # Each expert's share of the tokens in each block, for TensorBoard, and the smallest and largest
                # of them on the line: 12.5% each, with 8 experts, when they're even.
                shares = torch.stack([b.mlp.load.float() / b.mlp.load.sum() for b in model.blocks]).cpu()
                for layer, row in enumerate(shares):
                    for e, share in enumerate(row):
                        writer.add_scalar(f"experts/layer{layer}/expert{e}", share.item(), step)
                line += f" | experts {shares.min().item():.1%} to {shares.max().item():.1%}"
            print(line, flush=True)
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
    # --kv-heads N: grouped-query attention, with N heads of keys and values.
    kv_heads = int(sys.argv[sys.argv.index("--kv-heads") + 1]) if "--kv-heads" in sys.argv else None

    def option(name, default, kind=int):
        return kind(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

    # --moe N: a mixture of N experts in each block, each token going to --top-k of them (2), balanced by --balance:
    # loss (the default), bias, or none. --steps N ends the run after N steps instead of the minutes.
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 10
    main(minutes, rope="--rope" in sys.argv, bpe="--bpe" in sys.argv,
         diffusion="--diffusion" in sys.argv, init=init, modern="--modern" in sys.argv, kv_heads=kv_heads,
         hyena="--hyena" in sys.argv, experts=option("--moe", 0), top_k=option("--top-k", 2),
         balance=option("--balance", "loss", str), steps=option("--steps", None))
