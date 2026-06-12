"""Stage 1: download data, build the tokenizer, pretrain the GPT.

Pretraining is plain next-token prediction on TinyStories. Loss is
cross-entropy on the next token, so it starts near ln(vocab) ~ 9.2 (uniform
guessing) and falls toward ~2 as the model learns English.

Produces:
  data/*.txt, data/*.bin   (raw + tokenized corpora; downloads are skipped
                            when the files already exist)
  ckpt_base.pt             (weights + config + vocab via save_checkpoint)

Run:  python pretrain.py [minutes]      (default 10)
"""

import os
import sys
import time
from collections import Counter

import numpy as np
import requests
import torch

from base_model import (CKPT_BASE, DATA_DIR, EOT, SPECIALS, GPT, GPTConfig,
                        WordTokenizer, device, pretokenize, sample_story,
                        save_checkpoint)

"""# Data"""

HF = "https://huggingface.co/datasets/roneneldan"
DOWNLOADS = [  # (url, filename, capped?)
    (f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt", "pretrain_train.txt", True),
    (f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt", "pretrain_valid.txt", False),
    (f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt", "instruct_train.txt", True),
    (f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt", "instruct_valid.txt", False),
]

MAX_TRAIN_BYTES = 80 * 1024 * 1024  # ~80 MB per large train file


# Download any corpus file that isn't on disk yet. Idempotent, so the later
# stages (sft/lora/dpo/eval) call this too and only pay for what's missing.
def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for url, name, capped in DOWNLOADS:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            continue
        print(f"downloading {name} ...", flush=True)
        with requests.get(url, stream=True, timeout=60) as r:
            buf = bytearray()
            for chunk in r.iter_content(chunk_size=1 << 20):
                buf.extend(chunk)
                if capped and len(buf) >= MAX_TRAIN_BYTES:
                    break

        text = buf.decode("utf-8", errors="ignore")
        if capped:
            text = text[: text.rfind(EOT) + len(EOT)]  # end on a story boundary

        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{name}: {len(text) / 1e6:.1f} MB, ~{text.count(EOT)} stories", flush=True)


"""# Tokenizer"""

MIN_FREQ = 2
VOCAB_SIZE = 10_000

# The vocab is also written here (only when it changes), so encode_file can
# tell when an existing .bin was produced under a different vocab.
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.txt")


# Count token frequencies across both training files and keep the top
# VOCAB_SIZE. Deterministic given the same data, but the canonical vocab is
# whatever gets saved into ckpt_base.pt - later stages load it from there.
def build_tokenizer():
    files = [
        os.path.join(DATA_DIR, "pretrain_train.txt"),
        os.path.join(DATA_DIR, "instruct_train.txt"),
    ]

    # EOT is stripped so document boundaries don't pollute the counts.
    counts = Counter()
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                counts.update(pretokenize(line.replace(EOT, " ")))

    # Vocab: specials first, then tokens by frequency (ties broken alphabetically
    # so the result is deterministic across runs).
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = list(SPECIALS) + [t for t, c in ranked if c >= MIN_FREQ][: VOCAB_SIZE - len(SPECIALS)]
    print(f"vocab size: {len(vocab)} (from {len(counts)} unique tokens)")

    # Rewrite vocab.txt only when the content actually changed: its mtime then
    # marks "vocab last changed", which encode_file compares .bin files against.
    blob = "\n".join(vocab)
    old = None
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, encoding="utf-8") as f:
            old = f.read()
    if old != blob:
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            f.write(blob)

    return WordTokenizer(vocab)


"""# Encode Data"""


def encode_file(src_path, dst_path, tokenizer):
    # skip when the .bin is already newer than its source text AND the vocab -
    # a token .bin is stale if either of its inputs changed
    if os.path.exists(dst_path) and os.path.getmtime(dst_path) >= max(
            os.path.getmtime(src_path), os.path.getmtime(VOCAB_PATH)):
        return
    with open(src_path, encoding="utf-8") as f:
        text = f.read()
    arr = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    arr.tofile(dst_path)
    print(f"{os.path.basename(src_path)}: {len(arr):,} tokens -> "
          f"{os.path.basename(dst_path)} ({arr.nbytes / 1e6:.1f} MB)", flush=True)


"""# Train"""


# Returns a function that yields random (input, target) windows from a .bin
# token file. memmap keeps the corpus on disk; each batch touches only
# batch_size * block_size of it.
def make_batch_fn(bin_path, block_size, batch_size, device):
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def get_batch():
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        # target = same window shifted one right: position t learns token t+1
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    return get_batch


# Average loss over many batches with dropout off; a single batch is too noisy
# to compare train vs val.
@torch.no_grad()
def estimate_loss(model, batch_fns, iters=20):
    model.eval()
    out = {}
    for split, fn in batch_fns.items():
        out[split] = sum(model(*fn())[1].item() for _ in range(iters)) / iters
    model.train()
    return out


def main(minutes=10.0):
    torch.manual_seed(42)
    np.random.seed(42)

    ensure_data()
    tok = build_tokenizer()

    encode_file(os.path.join(DATA_DIR, "pretrain_train.txt"), os.path.join(DATA_DIR, "pretrain_train.bin"), tok)
    encode_file(os.path.join(DATA_DIR, "pretrain_valid.txt"), os.path.join(DATA_DIR, "pretrain_val.bin"), tok)

    cfg = GPTConfig(vocab_size=len(tok.id_to_token))
    model = GPT(cfg).to(device)
    print(f"model parameters: {model.num_params():,}")

    batch_size = 48
    get_train_batch = make_batch_fn(os.path.join(DATA_DIR, "pretrain_train.bin"), cfg.block_size, batch_size, device)
    get_val_batch = make_batch_fn(os.path.join(DATA_DIR, "pretrain_val.bin"), cfg.block_size, batch_size, device)
    batch_fns = {"train": get_train_batch, "val": get_val_batch}

    lr = 6e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)  # beta2 below the 0.999 default - standard for noisy LM gradients
    sched_iters = 8000   # cosine reaches its floor here

    # Decay the weight matrices (dim >= 2), not biases/LayerNorm params - decaying
    # those toward zero fights normalization instead of regularizing.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ], lr=lr, betas=betas)

    # LR schedule: linear warmup, then cosine decay.
    # Warmup: Adam's per-parameter step sizes are garbage for the first ~100 steps
    # (moment estimates are still mostly zeros), so ramp the lr up from ~0.
    # Cosine: decay smoothly to lr/10 so late steps make small, careful updates.
    min_lr = lr / 10
    warmup = 100
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-8, total_iters=warmup),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sched_iters - warmup, eta_min=min_lr),
    ], milestones=[warmup])

    eval_interval = 500
    t0 = last_log = time.time()
    deadline = t0 + minutes * 60
    it = tok_seen = 0

    # The classic training loop: sample batch -> forward (loss) -> backward
    # (gradients) -> clip -> optimizer step -> lr step. Wall-clock bounded;
    # loss should fall from ~9.2 (uniform) toward ~2.
    while time.time() < deadline:
        x, y = get_train_batch()
        opt.zero_grad(set_to_none=True)  # gradients accumulate by default - clear last step's
        _, loss = model(x, y)            # forward: mean next-token cross-entropy over B*T positions
        loss.backward()                  # backward: dL/dp for every parameter
        # cap total gradient norm at 1.0 - a rare bad batch can't blow up the weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()        # AdamW update
        scheduler.step()  # then advance lr (order matters)
        it += 1
        tok_seen += x.numel()

        if it % 50 == 0:
            now = time.time()
            tps = 50 * batch_size * cfg.block_size / (now - last_log)
            last_log = now
            print(f"iter {it:5d} | loss {loss.item():.3f} | lr {opt.param_groups[0]['lr']:.2e} | "
                  f"{tps:,.0f} tok/s | {(now - t0) / 60:.1f} min", flush=True)

        if it % eval_interval == 0:
            losses = estimate_loss(model, batch_fns)
            print(f"  >> eval: train {losses['train']:.3f}  val {losses['val']:.3f}", flush=True)
            print("  sample:", sample_story(model, tok), flush=True)  # watch fluency emerge
            save_checkpoint(CKPT_BASE, model, tok, iters=it)

    save_checkpoint(CKPT_BASE, model, tok, iters=it)
    print(f"\nDONE  iters={it}  tok_seen={tok_seen:,}  time={(time.time() - t0) / 60:.1f} min")
    print(f"saved {CKPT_BASE}")

    for i in range(3):
        print(f"\n--- sample {i} ---\n"
              f"{sample_story(model, tok, 'Once upon a time', n=160, eos_id=tok.eot_id)}")


if __name__ == "__main__":
    main(minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 10.0)
