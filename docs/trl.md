# GRPO, with Hugging Face's TRL

[GRPO](grpo.md) was written by hand in `grpo.py`. [TRL](https://huggingface.co/docs/trl), Hugging
Face's library for training language models after pretraining, has a GRPO trainer of its own, used on
models with billions of parameters. [`trl_grpo.py`](https://github.com/nenuadrian/mvGPT/blob/main/trl_grpo.py)
gives it our fine-tuned model and our reward, first with `grpo.py`'s settings and then with TRL's
defaults, and measures what it trained with our own scripts.

## Installing it

TRL is in `requirements.txt`, from version 1.17:

```bash
uv pip install -r requirements.txt
```

With our requirements, uv adds TRL and Hugging Face's `datasets`, with the packages `datasets` needs to
download and cache files, and leaves transformers and PyTorch as they were. TRL 1.17 has no PPO
trainer: its trainers are GRPO and RLOO, which have the model write its own answers as it trains, SFT,
DPO and KTO, which learn from text written beforehand, a reward model's and distillation's.

## Our model, as a GPT-2, and back

TRL trains the models of the transformers library. Ours has GPT-2's layout (learned positions,
LayerNorm, GELU, biases), so `hub.py`'s `to_gpt2` (from [sharing the model](deploy.md)) saves it as a
`GPT2LMHeadModel`, and `to_hf_tokenizer` its tokenizer. `from_gpt2` is the way back: a GPT-2's
weights into our model, so that `evaluate.py`, `sample.py` and `grpo.py`'s measurement can measure
what TRL trained. `tests/test_hub.py` checks that a model that goes to GPT-2 and back comes back with
each of its weights.

```bash
python -m pytest tests/test_hub.py
```

## The dataset and the reward

The prompts are the training instructions that ask for words, as text, in a `datasets.Dataset` with a
`prompt` column, which TRL tokenizes, and a `words` column. TRL passes each column other than the
prompt to the reward function by name, with the completions:

```python
def words_reward(completions, words, **kwargs):
    return [words_used(story, asked) / len(asked) for story, asked in zip(completions, words)]
```

A `TrainerCallback` prints a line per rollout as `grpo.py` does, and each ten rollouts copies TRL's
weights into our model with `from_gpt2` to measure the share of the words used on 50 held-out
instructions.

## Our settings, in GRPOConfig

| `grpo.py` | `GRPOConfig` | TRL's default |
| --- | --- | --- |
| 16 instructions x 8 stories a rollout | `num_generations=8`, `per_device_train_batch_size=16`, `steps_per_generation=8` | 8 stories a prompt |
| 2 passes over a rollout, in minibatches of 16 | `num_iterations=2` | 1 |
| PPO's clip, 0.2 | `epsilon=0.2` | 0.2 |
| A learning rate of $10^{-5}$, constant | `learning_rate=1e-5`, `lr_scheduler_type="constant"` | $10^{-6}$, decaying to 0 |
| A KL weight of 0.05, $e^{r} - r - 1$ per token | `beta=0.05` | 0, without a reference model |
| Each story's tokens averaged, then the stories | `loss_type="grpo"` | `"dapo"`, all the batch's tokens together |
| 200 tokens a story, a temperature of 1, no top-k | `max_completion_length=200`, `temperature=1.0` | 512 tokens |
| float32 | `bf16=False` | bfloat16 mixed precision |
| Activations kept for the backward pass | `gradient_checkpointing=False` | recomputed, to save memory |
| Groups whose stories all tie, counted | logged as `frac_reward_zero_std` | |

`python trl_grpo.py 32 matched` uses all of `grpo.py`'s settings. `python trl_grpo.py 32 defaults`
keeps our learning rate, schedule and story length, and takes TRL's defaults for the rest: no KL
penalty, [DAPO](https://arxiv.org/abs/2503.14476)'s loss, one pass, bfloat16 and gradient checkpointing. TRL's documentation of its
[loss types](https://huggingface.co/docs/trl/grpo_trainer#loss-types) compares GRPO's, DAPO's and
[Dr. GRPO](https://arxiv.org/abs/2503.20783)'s.

## A folder, for the reference

With a KL weight, `GRPOTrainer` makes the reference by loading the model again from its config's
name or path. A model built in memory has none, and the trainer stops with
`OSError: Can't load the configuration of ''`. So `trl_grpo.py` saves the GPT-2 and its tokenizer to
`checkpoints/sft-hf` and passes the folder, from which TRL loads the policy, the reference and the
tokenizer. With TRL's defaults, there's no reference to load.

```bash
python trl_grpo.py 32 matched    # grpo.py's settings, to checkpoints/trl/matched
python trl_grpo.py 32 defaults   # TRL's defaults, to checkpoints/trl/defaults
python evaluate.py checkpoints/sft/model.pt checkpoints/trl/matched/model.pt checkpoints/trl/defaults/model.pt
```

## Results

32 rollouts of each, from the fine-tuned model, on an Apple M4 Pro, measured by `evaluate.py`, next to
[GRPO](grpo.md)'s runs as that page measured them (its models weren't kept). The fine-tuned model measures as it
did there, to the digit.

| | Fine-tuned | `grpo.py`, KL weight 0.05 (episode 16) | TRL, matched | `grpo.py`, no KL (episode 16) | TRL, defaults |
| --- | --- | --- | --- | --- | --- |
| Requested words used | 27.3% | 50.0% | 44.7% | 58.0% | 57.3% |
| Stories that end within 200 tokens | 48% | 42% | 48% | 8% | 10% |
| Loss on plain stories | 2.38 | 2.46 | 2.47 | 2.56 | 2.53 |
| Loss on instructions' stories | 2.13 | 2.19 | 2.18 | 2.34 | 2.30 |

With `grpo.py`'s settings, TRL's rewards climb with ours for the first half of the run, then trail by a few
points, and it ends about 8 of the 150 requested words below `grpo.py`, with nearly the same loss on plain stories.
From one run of each, that gap could be the implementations or the seed; more seeds would settle it. With TRL's
defaults (no penalty, DAPO's loss, one pass, bfloat16), it lands next to `grpo.py` without a penalty, in stories
that run on and repeat the words asked for: reward hacking, as both runs without a penalty show.

## Speed

TRL was quicker than `grpo.py` on this Mac, and the difference is in writing the stories:

| Seconds per rollout, on an M4 Pro Mac mini | `grpo.py` | TRL, matched | TRL, defaults |
| --- | --- | --- | --- |
| Between evaluations | 10.0 | 7.1 | 4.5 |

Writing a rollout's 128 stories of 200 tokens: `grpo.py`'s 16 batches of 8 took 6.6 s, transformers'
`generate` with all 128 in one batch 1.6 s, and our `generate` with 128 copies of one prompt 1.2 s.

`grpo.py` writes a group at a time, as its prompts in a group share a length. To write a whole rollout in one
batch, `generate` would need padding and an attention mask for prompts of different lengths, which transformers'
`generate` has.
