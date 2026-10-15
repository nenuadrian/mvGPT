# Reinforcement learning, with PPO

DPO learns from pairs of stories written ahead of time. Reinforcement learning has the model write
stories as it trains, scores each one with a reward, and makes the stories that score well more
likely. Proximal policy optimisation (PPO) is the algorithm InstructGPT and the first ChatGPT were
trained with, against a reward model trained on people's preferences.

## The reward

Here the reward is scored by code, as in [DPO](dpo.md): the share of the words an instruction asks
for that a story uses, between 0 and 1. So the two methods aim at one target, and can be compared.

## The loop

[`ppo.py`](https://github.com/nenuadrian/mvGPT/blob/main/ppo.py) starts from the fine-tuned model,
with a frozen copy as the reference, and repeats:

1. **Rollout.** The policy writes 128 stories, for 16 groups of 8 instructions of one length,
   sampling from its full softmax, at a temperature of 1, with no top k, so the probabilities it
   trains on are the ones the tokens were drawn from. Each story is scored.
2. **Credit.** The reward goes on the story's last token. A critic, a linear layer on the channels
   after the final norm (read with a forward hook, so `model.py` doesn't change, and detached, so
   the critic learns from them without changing them), estimates the reward to come at each token.
   Generalised advantage estimation, with $\lambda = 1$, gives each token its advantage: the
   story's reward minus the critic's estimate, so a token is credited for doing better than
   expected. With $\lambda = 0.95$, the start of a 200-token story would get $0.95^{200}$ of the
   credit, almost nothing.
3. **Update.** Two passes over the rollout, in minibatches of 16 stories, with PPO's clipped
   objective: the ratio of a token's new probability to its old one, times its advantage, with the
   ratio clipped to [0.8, 1.2], so one rollout can't move the policy far. The critic learns to
   predict the rewards, and a KL penalty keeps the policy near the reference: per token,
   $e^{r} - r - 1$ with $r = \log \pi_\text{ref} - \log \pi_\theta$, which is never negative,
   times a weight, in the loss.

Putting the KL penalty in the reward instead, as InstructGPT did, made its noise larger than the
reward's own spread at this scale, and the policy learned little. A critic that shared its gradients
with the policy's channels collapsed it.

```bash
python ppo.py 10 0.05    # ten minutes, with a KL penalty weighted 0.05
python ppo.py 10 0.5     # and with ten times the weight
python evaluate.py checkpoints/sft/model.pt checkpoints/dpo/model.pt checkpoints/ppo/kl-0.05/model.pt checkpoints/ppo/kl-0.5/model.pt
```

## Results

Ten minutes of each, from the fine-tuned model, on an Apple M4 Pro, measured by `evaluate.py`:

| | Fine-tuned | DPO | PPO, KL weight 0.05 | PPO, KL weight 0.5 |
| --- | --- | --- | --- | --- |
| Requested words used | 27.3% | 63.3% | 44.7% | 28.0% |
| Stories that end within 200 tokens | 48% | 20% | 48% | 52% |
| Loss on plain stories | 2.38 | 2.87 | 2.52 | 2.40 |
| Loss on instructions' stories | 2.13 | 2.58 | 2.23 | 2.15 |

With a KL weight of 0.05, PPO takes the share of words used from 27% to 45% while it trains,
evaluation by evaluation, and its stories still end on their own, for a quarter of DPO's cost on
plain stories. With ten times the weight, the penalty holds the policy at the reference, and it
learns next to nothing: the weight sets how far the policy moves from the reference to raise the reward.

PPO writes its own stories as it trains, so it sees what the policy does now, where DPO learns from
pairs written once, by the fine-tuned model. It's also slower per update and has more to tune: the
KL weight, the clip, the learning rate, the critic. A reward written in code can be gamed:
with the KL penalty in the reward and a small weight, an earlier version learned to write
stories of about 17 tokens, which use the words and drift little in total.
