"""LoRA: parameter-efficient instruction tuning of the pretrained model.

Same task and data as sft.py, different mechanics. Instead of updating all
~15M weights, freeze the base model and learn a low-rank correction on the
attention q/v projections:

    h = W x + (alpha / r) * B (A x)

with A: (r, in) and B: (out, r), r << in. B starts at zero, so at step 0 the
adapted model is exactly the base model; training only moves the small A/B
matrices (~0.4% of the parameters here). The checkpoint stores just the
adapters - reconstruct the full model as base checkpoint + adapter checkpoint.

Loads ckpt_base.pt (from pretrain.py), produces ckpt_lora.pt.

Run:  python lora.py [minutes]      (default 5)
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from base_model import (CKPT_BASE, CKPT_LORA, DATA_DIR, device,
                        load_checkpoint, sample_story)
from pretrain import ensure_data
from sft import DEMO_PROMPT, build_examples, estimate_sft_loss, make_batch

# Which projections get adapters. q and v is the classic LoRA-paper choice;
# adding k/o/MLP buys little at this scale.
TARGETS = ("q_proj", "v_proj")


# Wraps a frozen nn.Linear and adds the trainable low-rank path.
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base                 # the pretrained Linear, kept frozen
        self.r, self.alpha = r, alpha
        self.scale = alpha / r           # decouples the update size from r
        dev, dt = base.weight.device, base.weight.dtype
        # A gets a standard init, B starts at zero -> B@A = 0, adapter is a no-op
        # at step 0 and the model starts exactly at the pretrained weights.
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # x -> r dims -> out dims: two skinny matmuls instead of one full-rank update
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


# Freeze every base parameter, then swap the target projections for LoRA
# wrappers (whose A/B are the only trainable tensors left).
def add_lora(model, r=8, alpha=16, targets=TARGETS):
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks:
        for name in targets:
            setattr(block.attn, name, LoRALinear(getattr(block.attn, name), r, alpha))
    return model


# Only the adapter tensors - this is the whole LoRA checkpoint (~300 KB
# vs ~60 MB for full weights).
def lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


# Rebuild the tuned model from its two pieces: base checkpoint + adapters.
def load_lora_model(base_ckpt=CKPT_BASE, lora_ckpt=CKPT_LORA, map_location=None):
    model, tok, _ = load_checkpoint(base_ckpt, map_location)
    blob = torch.load(lora_ckpt, map_location=map_location or device)
    add_lora(model, r=blob["r"], alpha=blob["alpha"], targets=tuple(blob["targets"]))
    missing, unexpected = model.load_state_dict(blob["lora"], strict=False)
    assert not unexpected, unexpected  # every adapter tensor must find its slot
    return model, tok


# Fold each adapter into its base weight (W += scale * B@A) and put the plain
# Linear back: identical outputs, zero inference overhead, but no longer
# separable from the base.
def merge_lora(model):
    for block in model.blocks:
        for name in TARGETS:
            mod = getattr(block.attn, name)
            if isinstance(mod, LoRALinear):
                mod.base.weight.data += (mod.lora_B @ mod.lora_A) * mod.scale
                setattr(block.attn, name, mod.base)
    return model


def main(minutes=5.0):
    torch.manual_seed(42)
    np.random.seed(42)

    ensure_data()
    model, tok, _ = load_checkpoint(CKPT_BASE)
    cfg = model.cfg

    r, alpha = 8, 16
    add_lora(model, r=r, alpha=alpha)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    max_iters = 100_000
    batch_size = 48
    lr = 1e-3  # adapters start at zero and are tiny - they take a much hotter lr than full FT
    eval_interval = 250

    examples = build_examples(os.path.join(DATA_DIR, "instruct_train.txt"),
                              tok, cfg.block_size, tok.eot_id, limit=40_000)
    val_examples = build_examples(os.path.join(DATA_DIR, "instruct_valid.txt"),
                                  tok, cfg.block_size, tok.eot_id, limit=5000)
    print(f"LoRA SFT examples: {len(examples):,}  (held-out val {len(val_examples):,})", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def save(step):
        torch.save({"lora": lora_state_dict(model), "r": r, "alpha": alpha,
                    "targets": list(TARGETS), "iters": step}, CKPT_LORA)

    model.train()
    t0 = time.time()
    deadline = t0 + minutes * 60
    it = 0

    # Identical loop to sft.py - the only difference is which parameters the
    # optimizer holds and what the gradients flow into.
    while it < max_iters and time.time() < deadline:
        idxs = torch.randint(len(examples), (batch_size,)).tolist()
        x, y = make_batch(examples, idxs, tok.pad_id, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        it += 1

        if it % 50 == 0:
            print(f"iter {it:5d} | loss {loss.item():.3f} | {(time.time() - t0) / 60:.1f} min", flush=True)

        if it % eval_interval == 0:
            val_loss = estimate_sft_loss(model, val_examples, batch_size, tok.pad_id)
            print(f"  >> eval iter {it}: val_loss {val_loss:.3f}", flush=True)
            story = sample_story(model, tok, DEMO_PROMPT, n=120, eos_id=tok.eot_id)
            print("  --- sample story ---\n  " + story.replace("\n", "\n  "), flush=True)
            save(it)

    save(it)
    print(f"\nDONE  iters={it}  time={(time.time() - t0) / 60:.1f} min")
    print(f"saved {CKPT_LORA} (adapters only)")

    # Round-trip demo: rebuild from ckpt_base.pt + ckpt_lora.pt, then merge.
    print("\n--- reloaded (base ckpt + adapters) ---")
    re_model, re_tok = load_lora_model()
    torch.manual_seed(0)
    print(sample_story(re_model, re_tok, DEMO_PROMPT, n=120, eos_id=re_tok.eot_id))

    print("\n--- merged (adapters folded into base weights; same model) ---")
    merge_lora(re_model)
    torch.manual_seed(0)
    print(sample_story(re_model, re_tok, DEMO_PROMPT, n=120, eos_id=re_tok.eot_id))


if __name__ == "__main__":
    main(minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 5.0)
