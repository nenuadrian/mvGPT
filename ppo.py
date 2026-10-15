import copy
import os
import sys
import textwrap
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from data import DATA_DIR
from dpo import DEMO, words_rate
from evaluate import load_instructions, words_used
from model import pick_device
from preferences import GROUP, MAX_TOKENS, load_asks
from sample import write

ROLLOUT = 16  # groups of instructions per rollout: 16 x 8 = 128 stories
EPOCHS = 2  # passes over each rollout
MINIBATCH = 16
CLIP = 0.2  # how far the probability of a token may move in a single update
LR = 1e-5
EVAL_EVERY = 10  # rollouts


class ValueHead(nn.Module):
    # The critic: from each position's channels, a guess at the reward the story is going to get.
    def __init__(self, n_embd):
        super().__init__()
        self.proj = nn.Linear(n_embd, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden):
        return self.proj(hidden).squeeze(-1)


def token_log_probs(logits, seq):
    # Position t predicts token t + 1, so the logits drop their last position and the tokens their first.
    return F.log_softmax(logits[:, :-1], dim=-1).gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rollout(policy, tokenizer, by_length, rng):
    device = next(policy.parameters()).device
    lengths = [length for length, asks in by_length.items() if len(asks) >= GROUP]
    stories = []
    for _ in range(ROLLOUT):
        length = lengths[rng.integers(len(lengths))]
        group = [by_length[length][i] for i in rng.choice(len(by_length[length]), GROUP, replace=False)]
        prompts = torch.tensor([ids for ids, _ in group], device=device)
        # No top k: PPO compares probabilities with the ones the tokens were sampled from, in full.
        out = policy.generate(prompts, MAX_TOKENS, temperature=1.0)
        for (ids, words), story in zip(group, out[:, length:].tolist()):
            story = story[: story.index(tokenizer.eot_id) + 1] if tokenizer.eot_id in story else story
            reward = words_used(tokenizer.decode(story), words) / len(words)
            stories.append((ids, story, reward))
    return stories


def pack(stories, pad_id, device):
    longest = max(len(ids) + len(story) for ids, story, _ in stories)
    seq = torch.full((len(stories), longest), pad_id, dtype=torch.long)
    mask = torch.zeros(len(stories), longest - 1)
    for row, (ids, story, _) in enumerate(stories):
        seq[row, : len(ids) + len(story)] = torch.tensor(ids + story)
        # On the shifted frame, the positions that predict the story's tokens.
        mask[row, len(ids) - 1 : len(ids) + len(story) - 1] = 1
    return seq.to(device), mask.to(device)


def advantages(rewards, values, mask, lam=1.0):
    # Generalized advantage estimation, from the last token back: how much better each token turned
    # out than the critic expected. Where the mask ends, so does the story, with no value after it.
    adv = torch.zeros_like(rewards)
    running = torch.zeros(rewards.size(0), device=rewards.device)
    for t in reversed(range(rewards.size(1))):
        more = mask[:, t + 1] if t + 1 < rewards.size(1) else torch.zeros_like(running)
        next_value = values[:, t + 1] if t + 1 < rewards.size(1) else torch.zeros_like(running)
        delta = rewards[:, t] + more * next_value - values[:, t]
        running = delta + lam * more * running
        adv[:, t] = running
    return adv * mask, (adv + values) * mask


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()


def main(minutes, kl):
    # kl weighs the penalty for drifting from the reference; each value gets a folder of its own.
    run_dir = os.path.join(CHECKPOINTS, "ppo", f"kl-{kl}")
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    device = pick_device()
    policy, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "sft", "model.pt"), device)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    policy.eval()  # dropout off, so the probabilities at the update match the rollout's
    critic = ValueHead(policy.config.n_embd).to(device)
    # A hook keeps the channels after the final norm, which the critic reads, without changing model.py.
    hidden = {}
    policy.norm.register_forward_hook(lambda module, inputs, output: hidden.update(h=output))
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(critic.parameters()), lr=LR, weight_decay=0.0)
    by_length = load_asks(os.path.join(DATA_DIR, "instruct_train.txt"), tokenizer, policy.config.block_size - MAX_TOKENS)
    instructions = load_instructions(os.path.join(DATA_DIR, "instruct_valid.txt"), 50)
    writer = SummaryWriter(os.path.join(RUNS, time.strftime(f"ppo-kl-{kl}-%Y%m%d-%H%M%S")))

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
        stories = rollout(policy, tokenizer, by_length, rng)
        seq, mask = pack(stories, tokenizer.pad_id, device)
        reward = torch.tensor([r for _, _, r in stories], device=device)
        with torch.no_grad():
            old = token_log_probs(policy(seq)[0], seq)
            values = critic(hidden["h"][:, :-1])
            ref = token_log_probs(reference(seq)[0], seq)
            # The reward goes on each story's last token, and the advantages spread it back over the story.
            rewards = torch.zeros_like(old)
            last = (mask * torch.arange(mask.size(1), device=device)).argmax(dim=1)
            rewards[torch.arange(len(stories)), last] = reward
            adv, returns = advantages(rewards, values * mask, mask)
            adv = (adv - adv[mask.bool()].mean()) / (adv[mask.bool()].std() + 1e-8) * mask

        for _ in range(EPOCHS):
            for rows in torch.randperm(len(stories)).split(MINIBATCH):
                rows = rows.to(device)
                logits, _ = policy(seq[rows])
                new = token_log_probs(logits, seq[rows])
                ratio = torch.exp(new - old[rows])
                # PPO's clipped objective: a token's probability gains nothing from moving past the clip.
                surrogate = torch.min(ratio * adv[rows], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[rows])
                policy_loss = -masked_mean(surrogate, mask[rows])
                # Detached: the critic learns from the policy's channels without changing them.
                value_loss = masked_mean((critic(hidden["h"][:, :-1].detach()) - returns[rows]) ** 2, mask[rows])
                # The KL from the reference, estimated per token, as e^(r) - r - 1 with r = log ref - log policy,
                # which is never negative, and is zero where the two agree.
                r = ref[rows] - new
                kl_loss = masked_mean(torch.exp(r) - r - 1, mask[rows])
                loss = policy_loss + 0.5 * value_loss + kl * kl_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 1.0)
                optimizer.step()
        step += 1

        # The KL of each story from the reference: the sum of its tokens' log-ratios, from the rollout.
        drift = ((old - ref) * mask).sum(dim=1).mean().item()
        writer.add_scalar("rollout/reward", reward.mean().item(), step)
        writer.add_scalar("rollout/kl", drift, step)
        print(f"rollout {step:>4} | reward {reward.mean().item():.3f} | kl {drift:.2f} | "
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
