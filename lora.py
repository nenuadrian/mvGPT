import math
import os
import sys
import textwrap
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from data import DATA_DIR
from model import pick_device
from sample import write
from sft import DEMO, EVAL_EVERY, build_examples, estimate_loss, random_batch

RUN_DIR = os.path.join(CHECKPOINTS, "lora")
RANK = 8
ALPHA = 16
LR = 1e-3  # higher than fine-tuning's, as the adapters start from nothing and are few
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    # A frozen linear layer, plus a trainable low-rank update: W x + (alpha / r) B A x.
    def __init__(self, base, rank=RANK, alpha=ALPHA):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B starts at zero, so the adapted model starts as the pretrained one.
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def add_lora(model):
    model.requires_grad_(False)
    for block in model.blocks:
        for parent in (block.attn, block.mlp):
            for name in TARGETS:
                if hasattr(parent, name):
                    setattr(parent, name, LoRALinear(getattr(parent, name)))
    return model


@torch.no_grad()
def merge(model):
    # Fold each update into its weight, W + (alpha / r) B A, and put the plain layer back.
    for block in model.blocks:
        for parent in (block.attn, block.mlp):
            for name in TARGETS:
                layer = getattr(parent, name, None)
                if isinstance(layer, LoRALinear):
                    layer.base.weight += layer.scale * layer.B @ layer.A
                    setattr(parent, name, layer.base)
    return model


def main(minutes):
    torch.manual_seed(1337)
    device = pick_device()
    model, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "pretrain", "model.pt"), device)
    add_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"trainable: {n_trainable:,} of {total:,} parameters, {n_trainable / total:.2%}", flush=True)
    block_size = model.config.block_size
    train = build_examples(os.path.join(DATA_DIR, "instruct_train.txt"), tokenizer, block_size)
    valid = build_examples(os.path.join(DATA_DIR, "instruct_valid.txt"), tokenizer, block_size, limit=5000)
    optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    writer = SummaryWriter(os.path.join(RUNS, time.strftime("lora-%Y%m%d-%H%M%S")))

    def evaluate(step):
        losses = {"train": estimate_loss(model, train, tokenizer.pad_id),
                  "valid": estimate_loss(model, valid, tokenizer.pad_id)}
        story = write(model, tokenizer, DEMO, max_new_tokens=120, with_prompt=False)
        writer.add_scalar("loss/train", losses["train"], step)
        writer.add_scalar("loss/valid", losses["valid"], step)
        writer.add_text("sample", story, step)
        writer.flush()
        print(f"step {step:>5} | train {losses['train']:.3f} | valid {losses['valid']:.3f}", flush=True)
        print(textwrap.indent(textwrap.fill(story, width=86), "  "), flush=True)

    budget = minutes * 60
    start = time.time()
    step = 0
    evaluate(step)
    while time.time() - start < budget:
        x, y = random_batch(train, tokenizer.pad_id, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        step += 1
        if step % 50 == 0:
            writer.add_scalar("loss/batch", loss.item(), step)
            print(f"step {step:>5} | loss {loss.item():.3f} | {(time.time() - start) / 60:.1f} min", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step)

    if step % EVAL_EVERY != 0:
        evaluate(step)
    writer.close()
    # AdamW keeps two numbers per trainable parameter: its running averages of the gradient and its square.
    state = sum(t.numel() * t.element_size() for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t))
    print(f"optimizer state: {state / 1e6:.1f} MB, against {total * 8 / 1e6:.0f} MB for all the parameters", flush=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    adapters = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    torch.save({"adapters": adapters, "rank": RANK, "alpha": ALPHA}, os.path.join(RUN_DIR, "adapters.pt"))
    print(f"adapters: {os.path.getsize(os.path.join(RUN_DIR, 'adapters.pt')) / 1e6:.2f} MB", flush=True)
    # Merged, it's a plain model again, which sample.py and evaluate.py load like any other.
    save_checkpoint(os.path.join(RUN_DIR, "model.pt"), merge(model), tokenizer, step=step)
    print(f"done: {step:,} steps, {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
