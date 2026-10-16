import copy
import os
import sys
import textwrap
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from data import DATA_DIR
from dpo import DEMO, words_rate
from evaluate import load_instructions, words_used
from model import pick_device
from ppo import CLIP, EPOCHS, LR, MINIBATCH, pack, token_log_probs
from preferences import MAX_TOKENS, load_asks
from sample import write

PROMPTS = 16  # instructions per rollout
GROUP = 8  # stories per instruction: 16 x 8 = 128 stories, as many as a rollout of PPO's
EVAL_EVERY = 10  # rollouts


@torch.no_grad()
def rollout(policy, tokenizer, asks, rng):
    device = next(policy.parameters()).device
    stories = []
    for i in rng.choice(len(asks), PROMPTS, replace=False):
        ids, words = asks[i]
        # A group: one instruction, written for eight times, so its stories can be scored against each other.
        out = policy.generate(torch.tensor([ids] * GROUP, device=device), MAX_TOKENS, temperature=1.0)
        for story in out[:, len(ids) :].tolist():
            story = story[: story.index(tokenizer.eot_id) + 1] if tokenizer.eot_id in story else story
            stories.append((ids, story, words_used(tokenizer.decode(story), words) / len(words)))
    return stories


def group_advantages(reward):
    # In place of a critic: each story's reward against its group's, in spreads above the group's mean.
    groups = reward.view(-1, GROUP)
    adv = (groups - groups.mean(dim=1, keepdim=True)) / (groups.std(dim=1, keepdim=True) + 1e-4)
    # A group whose stories all tie has nothing to tell them apart: zeros, and it teaches nothing.
    tied = (groups == groups[:, :1]).all(dim=1)
    return adv.masked_fill(tied[:, None], 0.0).view(-1), tied


def main(minutes, kl):
    run_dir = os.path.join(CHECKPOINTS, "grpo", f"kl-{kl}")
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    device = pick_device()
    policy, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "sft", "model.pt"), device)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    policy.eval()  # dropout off, so the probabilities at the update match the rollout's
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)
    by_length = load_asks(os.path.join(DATA_DIR, "instruct_train.txt"), tokenizer, policy.config.block_size - MAX_TOKENS)
    asks = [ask for group in by_length.values() for ask in group]
    instructions = load_instructions(os.path.join(DATA_DIR, "instruct_valid.txt"), 50)
    writer = SummaryWriter(os.path.join(RUNS, time.strftime(f"grpo-kl-{kl}-%Y%m%d-%H%M%S")))

    def evaluate(step):
        used = words_rate(policy, tokenizer, instructions)
        story = write(policy, tokenizer, DEMO, max_new_tokens=120, with_prompt=False)
        writer.add_scalar("held_out/words_used", used, step)
        writer.add_text("sample", story, step)
        writer.flush()
        print(f"rollout {step:>4} | words used {used:.1%}", flush=True)
        print(textwrap.indent(textwrap.fill(story, width=86), "  "), flush=True)
        save_checkpoint(os.path.join(run_dir, f"step-{step}.pt"), policy, tokenizer, step=step)

    budget = minutes * 60
    start = time.time()
    step = 0
    evaluate(step)
    while time.time() - start < budget:
        stories = rollout(policy, tokenizer, asks, rng)
        seq, mask = pack(stories, tokenizer.pad_id, device)
        reward = torch.tensor([r for _, _, r in stories], device=device)
        # One advantage per story, for each of its tokens.
        adv, tied = group_advantages(reward)
        with torch.no_grad():
            old = token_log_probs(policy(seq)[0], seq)
            ref = token_log_probs(reference(seq)[0], seq)

        for _ in range(EPOCHS):
            for rows in torch.randperm(len(stories)).split(MINIBATCH):
                rows = rows.to(device)
                new = token_log_probs(policy(seq[rows])[0], seq[rows])
                ratio = torch.exp(new - old[rows])
                a = adv[rows].unsqueeze(1)
                # PPO's clipped objective, and the KL from the reference per token, as in ppo.py.
                surrogate = torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
                r = ref[rows] - new
                per_token = (surrogate - kl * (torch.exp(r) - r - 1)) * mask[rows]
                # A mean over each story's tokens, then over the stories: a story counts once, however long.
                loss = -(per_token.sum(dim=1) / mask[rows].sum(dim=1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
        step += 1

        drift = ((old - ref) * mask).sum(dim=1).mean().item()
        ties = tied.sum().item()
        writer.add_scalar("rollout/reward", reward.mean().item(), step)
        writer.add_scalar("rollout/kl", drift, step)
        writer.add_scalar("rollout/tied_groups", ties, step)
        print(f"rollout {step:>4} | reward {reward.mean().item():.3f} | kl {drift:.2f} | tied {ties:>2}/{PROMPTS} | "
              f"{(time.time() - start) / 60:.1f} min", flush=True)
        if step % EVAL_EVERY == 0:
            evaluate(step)

    if step % EVAL_EVERY != 0:
        evaluate(step)
    writer.close()
    save_checkpoint(os.path.join(run_dir, "model.pt"), policy, tokenizer, step=step)
    print(f"done: {step:,} rollouts, {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10, float(sys.argv[2]) if len(sys.argv) > 2 else 0.05)
