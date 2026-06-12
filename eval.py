"""Stage comparison: pretrain vs SFT (vs LoRA) vs DPO.

Loads each stage's checkpoint from disk - no training state needed, so this
runs standalone any time after the stages you want to compare have finished.
Missing checkpoints are skipped with a note (lora is optional).

Run:  python eval.py
"""

import os

import numpy as np
import torch
from torch.nn import functional as F

from base_model import (CKPT_BASE, CKPT_DPO, CKPT_LORA, CKPT_SFT, DATA_DIR,
                        device, load_checkpoint, sample_story)
from dpo import (build_eval_pool, load_prompts, mean_reward,
                 required_words_reward, sentiment_reward, words)
from pretrain import ensure_data

GEN_LEN = 80
TOP_K = 50
REWARD_MODE = "both"


# ---- 4) next-token probability probe (no sampling noise) ----
@torch.no_grad()
def next_word_probs(model, tok, prefix, candidates):
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    ctx = torch.tensor([tok.encode(prefix)], device=dev)
    logits, _ = model(ctx)
    probs = F.softmax(logits[0, -1], dim=-1)
    model.train(was_training)
    return {w: probs[tok.token_to_id[w]].item() for w in candidates}


def load_models():
    """(name, model) for every stage whose checkpoint exists, plus a tokenizer."""
    models, tok = [], None
    for name, path in [("pretrain", CKPT_BASE), ("sft", CKPT_SFT)]:
        if os.path.exists(path):
            m, tok, _ = load_checkpoint(path)
            models.append((name, m.eval()))
        else:
            print(f"[skip] {name}: {path} not found")

    # LoRA is stored as adapters only; rebuild on top of the base checkpoint.
    if os.path.exists(CKPT_LORA) and os.path.exists(CKPT_BASE):
        from lora import load_lora_model
        m, tok = load_lora_model()
        models.append(("lora", m.eval()))
    else:
        print(f"[skip] lora: {CKPT_LORA} not found")

    if os.path.exists(CKPT_DPO):
        m, tok, _ = load_checkpoint(CKPT_DPO)
        models.append(("dpo", m.eval()))
    else:
        print(f"[skip] dpo: {CKPT_DPO} not found")
    return models, tok


def main():
    torch.manual_seed(42)
    models, tok = load_models()
    if not models:
        print("no checkpoints found - run pretrain.py (then sft.py / lora.py / dpo.py) first")
        return

    # ---- 1) instruction following on a hard prompt (unusual words, checkable) ----
    hard_prompt = ("Features: Dialogue\nWords: spoon, bird, brave\n"
                   "Summary: A brave bird helps a friend find a lost spoon.\nStory:")
    print("=" * 30, "hard instruction prompt", "=" * 30)
    for name, m in models:
        s = sample_story(m, tok, hard_prompt, n=120, eos_id=tok.eot_id)
        stopped = len(words(s)) < 110  # eos before token cap -> learned to stop
        print(f"\n--- {name} | words {required_words_reward(hard_prompt, s):.2f} | "
              f"sentiment {sentiment_reward(s):+.2f} | stopped {stopped} ---\n{s}")

    # ---- 2) sad instruction: SFT obeys, DPO drags toward happy ----
    sad_prompt = "Summary: A dog loses his toy and cries.\nStory:"
    print("\n" + "=" * 30, "sad instruction prompt", "=" * 30)
    for name, m in models:
        s = sample_story(m, tok, sad_prompt, n=120, eos_id=tok.eot_id)
        print(f"\n--- {name} | sentiment {sentiment_reward(s):+.2f} ---\n{s}")

    # ---- 3) mean reward over a fixed eval pool: one number per stage, one trend ----
    ensure_data()  # needs instruct_valid.txt for the prompt pool
    rng = np.random.default_rng(0)  # same seed as dpo.py -> same eval pool
    cfg = models[0][1].cfg
    buckets = load_prompts(os.path.join(DATA_DIR, "instruct_valid.txt"),
                           tok, cfg.block_size - GEN_LEN - 1, limit=20_000)
    _, eval_byl = build_eval_pool(buckets, rng, size=64)

    print("\n" + "=" * 30, "mean reward (64 prompts)", "=" * 30)
    for name, m in models:
        print(f"{name:9s} {mean_reward(m, tok, eval_byl, GEN_LEN, TOP_K, REWARD_MODE):+.3f}",
              flush=True)

    print("\n" + "=" * 30, "P(next word) after 'The dog felt very'", "=" * 30)
    for name, m in models:
        p = next_word_probs(m, tok, "The dog felt very", ["happy", "sad", "scared", "excited"])
        print(f"{name:9s} " + "  ".join(f"{w} {v:.3f}" for w, v in p.items()))


if __name__ == "__main__":
    main()
