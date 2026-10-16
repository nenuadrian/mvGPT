# Reinforcement learning, with GRPO

[PPO](ppo.md) needs a critic to say what reward a story is expected to get. Group relative policy
optimisation (GRPO), from [DeepSeekMath](https://arxiv.org/abs/2402.03300) and used to train
[DeepSeek-R1](https://arxiv.org/abs/2501.12948), does without one: the model writes a group of
answers to each question, and each answer is scored against the rest of its group.

## The reward

The reward is PPO's: the share of the words an instruction asks for that a story uses, between 0 and
1, scored by code. So the two methods aim at one target, and can be compared.

## The loop

[`grpo.py`](https://github.com/nenuadrian/mvGPT/blob/main/grpo.py) starts from the fine-tuned model,
with a frozen copy as the reference, and repeats:

1. **Rollout.** The policy writes 8 stories for each of 16 instructions, 128 stories, as many as in
   a rollout of PPO's, sampling from its full softmax at a temperature of 1. Each story is scored.
2. **Advantages.** In place of a critic, a story's advantage is its reward against its group's:
   $\hat A_i = (r_i - \operatorname{mean}(r)) / \operatorname{std}(r)$, with $10^{-4}$ added to the
   standard deviation, and each of the story's tokens gets it. A group whose eight stories all score
   alike has nothing to tell them apart: its advantages are 0, and it teaches nothing. `grpo.py`
   counts those groups, as `tied`.
3. **Update.** PPO's two passes over the rollout, in minibatches of 16, with PPO's settings,
   imported from `ppo.py`: the clipped objective, with the ratio clipped to [0.8, 1.2], and a
   learning rate of $10^{-5}$. The KL from the reference goes in the objective per token, as
   $e^{r} - r - 1$ with $r = \log \pi_\text{ref} - \log \pi_\theta$, times a weight, as in
   DeepSeekMath's equation 3:

$$
J = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big[ \min\big(\rho_{i,t} \hat A_i,\;
\operatorname{clip}(\rho_{i,t}, 1 - \epsilon, 1 + \epsilon) \hat A_i\big) - \beta\, \text{KL}_{i,t} \Big]
$$

Each story's tokens are averaged first, then the stories, so a story counts once, however long it is.
[DAPO](https://arxiv.org/abs/2503.14476) changes that to an average over all the tokens of a group,
as `ppo.py` does, drops the KL penalty, and samples more questions in place of the groups that tie.
`grpo.py` follows DeepSeekMath, with the KL weight as an argument, so that 0 drops the penalty. The
reference stays either way, to measure how far the policy drifts.

```bash
python grpo.py 10 0.05   # ten minutes, with a KL penalty weighted 0.05, PPO's
python grpo.py 10 0      # and with no penalty, as DAPO does
python evaluate.py checkpoints/sft/model.pt checkpoints/ppo/kl-0.05/model.pt checkpoints/grpo/kl-0.05/model.pt checkpoints/grpo/kl-0.0/model.pt
```

`evaluate.py` names a run per KL weight with its method, `ppo-kl-0.05` and `grpo-kl-0.05`, as both
methods save to folders named after the weight.

## Results

Ten minutes of each, from the fine-tuned model, on an Apple M4, with PPO run again on that laptop,
measured by `evaluate.py`:

| | Fine-tuned | PPO, KL weight 0.05 | GRPO, KL weight 0.05 | GRPO, no KL |
| --- | --- | --- | --- | --- |
| Rollouts in ten minutes | | 37 | 32 | 36 |
| KL per story, last five rollouts | | 18.7 | 14.4 | 44.1 |
| Requested words used | 27.3% | 44.7% | 50.0% | 58.0% |
| Stories that end within 200 tokens | 48% | 30% | 42% | 8% |
| Loss on plain stories | 2.38 | 2.48 | 2.46 | 2.56 |
| Loss on instructions' stories | 2.13 | 2.20 | 2.19 | 2.34 |

With PPO's KL weight, GRPO lands close to PPO: a few more of the words, for a similar rise in the
loss. The 50 instructions ask for 150 words in all, so 5 points is about 8 words, from a single run
of each, which another seed could close or widen. Without the penalty, the policy drifts furthest
and uses the most words, in stories that mostly run to the end of their tokens: a longer story has
more chances to use a word, and the reward doesn't mind repeats. Both are reward hacking, as with
DPO and PPO.

Few groups tie with this reward: about 4% with the penalty and 5% without, over the runs, as eight
stories rarely all use the same number of the words asked for. In maths, where an answer is right or wrong, DAPO
found more and more questions with all their answers right as training went on.

## What's real, and what's scaled down

- The algorithm is DeepSeekMath's: a group of answers per question, the advantage against the
  group's mean and standard deviation, PPO's clipped objective, and a KL penalty per token, averaged
  per answer.
- DeepSeekMath's model had 7 billion parameters and wrote 64 answers per question, with a learning
  rate of $10^{-6}$, a KL weight of 0.04, and a reward model; DeepSeek-R1-Zero's reward came from
  rules, on whether the answer is right and in the format asked for. Ours has 14.6 million
  parameters, writes 8 stories per instruction, and counts words.
- DeepSeekMath makes a single update per rollout, so the ratio stays at 1 and the clip doesn't come
  into play; `grpo.py` keeps PPO's two passes, where it does.
- PPO's critic was a single linear layer here, so dropping it saves next to nothing; InstructGPT's
  was a model of 6 billion parameters, and at that size, not training or holding a second model is
  GRPO's saving.
