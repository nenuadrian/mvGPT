"""Stage 3: DPO (direct preference optimization) on top of the SFT model.

Build preference pairs by sampling k completions per prompt from the model,
scoring them with cheap programmatic rewards (sentiment, required words),
and pairing best vs worst. The DPO loss then pushes the policy to rank
chosen above rejected relative to a frozen reference copy of the SFT model.

Loads ckpt_sft.pt (from sft.py), produces ckpt_dpo.pt.

Run:  python dpo.py [minutes] [n_pairs]      (defaults 10, 2000)
"""

import copy
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from base_model import (CKPT_DPO, CKPT_SFT, DATA_DIR, EOT, device,
                        load_checkpoint, sample_story, save_checkpoint)
from pretrain import ensure_data
from sft import MARKER

"""## Rewards"""

"""Reward functions for the DPO stage. Each returns a float per story.

  - sentiment: positive words good, negative words bad, roughly [-1, 1]
  - required_words: fraction of the prompt's "Words:" list present in the story
"""

POSITIVE_WORDS = {
    "happy", "happily", "joy", "joyful", "smile", "smiled", "smiling", "laugh",
    "laughed", "laughing", "fun", "love", "loved", "loving", "kind", "kindly",
    "friend", "friends", "friendly", "play", "played", "playing", "good", "great",
    "wonderful", "nice", "proud", "excited", "cheer", "cheered", "hug", "hugged",
    "share", "shared", "sharing", "brave", "safe", "warm", "yay", "hooray",
    "best", "beautiful", "delighted", "glad", "grateful", "thank", "thanked",
}

NEGATIVE_WORDS = {
    "sad", "sadly", "cry", "cried", "crying", "scared", "scary", "afraid",
    "angry", "mad", "hurt", "hurts", "pain", "bad", "terrible", "awful", "fear",
    "fearful", "worried", "worry", "lonely", "alone", "dark", "monster", "die",
    "died", "dead", "broken", "lost", "lose", "cruel", "mean", "fight",
    "fought", "yell", "yelled", "scream", "screamed", "frown", "frowned", "sick",
}

WORD_RE = re.compile(r"[a-zA-Z']+")


def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def sentiment_reward(text):
    toks = words(text)
    if not toks:
        return 0.0
    score = sum(t in POSITIVE_WORDS for t in toks) - sum(t in NEGATIVE_WORDS for t in toks)
    # normalize by length, with a floor so short stories aren't over-rewarded
    return max(-1.0, min(1.0, score / max(8.0, len(toks) / 12)))


def required_words_reward(prompt_text, gen_text):
    m = re.search(r"Words:\s*(.+)", prompt_text)
    wanted = [w.strip().lower() for w in m.group(1).split(",") if w.strip()] if m else []
    if not wanted:
        return 0.0
    gen = set(words(gen_text))
    # prefix match so "jump" counts "jumped"/"jumping"
    hits = sum(any(g.startswith(w) for g in gen) for w in wanted)
    return hits / len(wanted)


def length_penalty(gen_text, target=70, span=60):
    return -min(1.0, abs(len(words(gen_text)) - target) / span)


def compute_reward(prompt_text, gen_text, mode="sentiment"):
    """mode in {"sentiment", "words", "both"}."""
    r = 0.1 * length_penalty(gen_text)
    if mode in ("sentiment", "both"):
        r += sentiment_reward(gen_text)
    if mode in ("words", "both"):
        r += 0.5 * required_words_reward(prompt_text, gen_text)
    return r


"""## Pairs Data"""


# Truncate ids at the first EOT, keeping it; the model's "stop" marker
def cut_at_eot(ids, eot):
    return ids[: ids.index(eot) + 1] if eot in ids else ids


# Build a buffer of (prompt_ids, chosen_ids, rejected_ids): sample k completions
# per prompt from the current policy, score them, pair best vs worst.
# Tied scores carry no preference signal, so those prompts are skipped.
@torch.no_grad()
def generate_pairs(model, tok, buckets, lengths, weights, rng, n_pairs, k,
                   gen_batch, gen_len, temperature, top_k, reward_mode, eot):
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    pairs, attempts = [], 0
    while len(pairs) < n_pairs and attempts < n_pairs * 6:
        plen = int(rng.choice(lengths, p=weights))
        bucket = buckets[plen]
        rows = [bucket[i] for i in rng.integers(0, len(bucket), size=gen_batch)]
        attempts += gen_batch

        prompts = torch.tensor([ids for _, ids in rows], device=dev)
        out = model.generate(prompts.repeat_interleave(k, dim=0),  # k samples per prompt
                             max_new_tokens=gen_len, temperature=temperature, top_k=top_k)
        comps = out[:, plen:].tolist()

        for b, (ptext, pids) in enumerate(rows):
            scored = sorted(
                (compute_reward(ptext, tok.decode(c), mode=reward_mode), c)
                for c in (cut_at_eot(comps[b * k + j], eot) for j in range(k))
            )
            (lo, worst), (hi, best) = scored[0], scored[-1]
            if hi - lo > 1e-6 and best and worst:
                pairs.append((pids, best, worst))
    model.train(was_training)
    return pairs[:n_pairs]


# Pack chosen and rejected into padded (seq, mask) tensors. mask=1 on completion
# tokens only, so prompt and padding drop out of the log-prob sums.
def make_dpo_batch(pairs, idxs, pad_id, device):
    rows = [pairs[i] for i in idxs]
    max_len = max(len(p) + max(len(c), len(r)) for p, c, r in rows)

    def pack(which):
        x = np.full((len(rows), max_len), pad_id, dtype=np.int64)
        m = np.zeros((len(rows), max_len), dtype=np.float32)
        for i, (p, c, r) in enumerate(rows):
            comp = c if which == "chosen" else r
            x[i, : len(p) + len(comp)] = p + comp
            m[i, len(p) : len(p) + len(comp)] = 1.0
        return torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)

    return (*pack("chosen"), *pack("rejected"))


# log pi(completion | prompt) per row: logits at position t predict token t+1,
# so gather targets seq[:, 1:] and sum where the (shifted) completion mask is on.
def completion_logprob(model, seq, comp_mask):
    logits, _ = model(seq)
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    tok_logp = torch.gather(logp, 2, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (tok_logp * comp_mask[:, 1:]).sum(dim=1)


"""## Prompts / eval pool"""


# Parse instruct docs into prompts ending at "Story:", grouped by token length.
# Same-length prompts can be stacked into one rectangular (B, T) tensor with no
# padding; lengths with <8 prompts are dropped as not worth a batch.
def load_prompts(path, tok, max_prompt_len, limit=None):
    with open(path, encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split(EOT) if d.strip()][:limit]

    buckets = {}
    for d in docs:
        i = d.find(MARKER)
        if i == -1:
            continue
        ptext = d[: i + len(MARKER)]
        pids = tok.encode(ptext)
        if 4 <= len(pids) <= max_prompt_len:
            buckets.setdefault(len(pids), []).append((ptext, pids))

    return {n: items for n, items in buckets.items() if len(items) >= 8}


# Draw a fixed pool of eval prompts and group them by token length so each
# group batches into one rectangular tensor.
def build_eval_pool(buckets, rng, size=64):
    pool = [p for items in buckets.values() for p in items]
    eval_prompts = [pool[i] for i in rng.integers(0, len(pool), size=size)]
    eval_byl = {}
    for ptext, pids in eval_prompts:
        eval_byl.setdefault(len(pids), []).append((ptext, pids))
    return eval_prompts, eval_byl


# Mean reward of the policy on the fixed eval prompts. Shared with eval.py.
@torch.no_grad()
def mean_reward(model, tok, eval_byl, gen_len, top_k, reward_mode):
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    rewards = []
    for plen, items in eval_byl.items():
        prompts = torch.tensor([ids for _, ids in items], device=dev)
        out = model.generate(prompts, max_new_tokens=gen_len, temperature=0.8, top_k=top_k)
        for row, (ptext, _) in enumerate(items):
            comp = cut_at_eot(out[row].tolist()[plen:], tok.eot_id)
            rewards.append(compute_reward(ptext, tok.decode(comp), mode=reward_mode))
    model.train(was_training)
    return float(np.mean(rewards)) if rewards else 0.0


"""## DPO Training"""


def main(minutes=10.0, n_pairs=2000):
    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(0)

    ensure_data()

    # Policy starts from the SFT weights; dropout off everywhere so log-probs
    # are deterministic. The frozen copy is the reference that anchors updates.
    model, tok, _ = load_checkpoint(CKPT_SFT)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    cfg = model.cfg
    batch_size = 16  # pairs per update

    # pair-generation knobs
    k = 4            # samples per prompt (best vs worst)
    gen_batch = 16   # prompts per generation call
    gen_len = 80
    temperature = 1.0
    top_k = 50
    regen_interval = 250  # steps between refreshing pairs from the current policy
    reward_mode = "both"  # choices: "sentiment", "words", "both"

    lr = 1e-5
    beta = 0.3  # how hard the margin is pushed relative to the reference
    eval_interval = 50

    max_prompt_len = cfg.block_size - gen_len - 1
    buckets = load_prompts(os.path.join(DATA_DIR, "instruct_valid.txt"),
                           tok, max_prompt_len, limit=20_000)

    # sample prompt lengths proportional to how many prompts each bucket holds
    lengths = list(buckets)
    weights = np.array([len(buckets[n]) for n in lengths], dtype=np.float64)
    weights /= weights.sum()

    eval_prompts, eval_byl = build_eval_pool(buckets, rng, size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    t0 = time.time()
    deadline = t0 + minutes * 60
    step = 0
    pairs = []

    while time.time() < deadline:
        if step % regen_interval == 0:
            print(f"generating {n_pairs} preference pairs ...", flush=True)
            pairs = generate_pairs(model, tok, buckets, lengths, weights, rng, n_pairs, k,
                                   gen_batch, gen_len, temperature, top_k, reward_mode, tok.eot_id)
            print(f"  got {len(pairs)} pairs", flush=True)

        idxs = rng.integers(0, len(pairs), size=batch_size).tolist()
        cs, cm, rs, rm = make_dpo_batch(pairs, idxs, tok.pad_id, device)

        pi_c = completion_logprob(model, cs, cm)
        pi_r = completion_logprob(model, rs, rm)
        with torch.no_grad():  # frozen reference
            ref_c = completion_logprob(ref, cs, cm)
            ref_r = completion_logprob(ref, rs, rm)

        # DPO: prefer chosen over rejected by the reference-anchored margin
        margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
        loss = -F.logsigmoid(margin).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 10 == 0:
            acc = (margin > 0).float().mean()  # fraction of pairs ranked correctly
            print(f"step {step:4d} | loss {loss.item():.4f} | pref_acc {acc.item():.2f} | "
                  f"margin {margin.mean().item():+.3f} | {(time.time() - t0) / 60:.1f} min", flush=True)

        if step % eval_interval == 0:
            r = mean_reward(model, tok, eval_byl, gen_len, top_k, reward_mode)
            print("  sample:", sample_story(model, tok, eval_prompts[0][0],
                                            n=gen_len, eos_id=tok.eot_id)[:300], flush=True)
            print(f"[eval] step {step}  mean_reward={r:+.3f}", flush=True)
            save_checkpoint(CKPT_DPO, model, tok, reward_mode=reward_mode, step=step)

    save_checkpoint(CKPT_DPO, model, tok, reward_mode=reward_mode, step=step)
    print(f"\nDONE  steps={step}", flush=True)
    print(f"saved {CKPT_DPO}")


if __name__ == "__main__":
    main(minutes=float(sys.argv[1]) if len(sys.argv) > 1 else 10.0,
         n_pairs=int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
