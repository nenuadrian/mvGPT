import os
import sys
import textwrap
import time

import torch
from datasets import Dataset
from transformers import ProgressCallback, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from checkpoint import CHECKPOINTS, RUNS, load_checkpoint, save_checkpoint
from data import DATA_DIR
from dpo import DEMO, words_rate
from evaluate import load_instructions, words_used
from grpo import EVAL_EVERY, GROUP, PROMPTS
from hub import from_gpt2, to_gpt2, to_hf_tokenizer
from model import pick_device
from ppo import CLIP, EPOCHS, LR, MINIBATCH
from preferences import MAX_TOKENS
from sample import write
from sft import instruction, parse
from tokenizer import EOT

STEPS = PROMPTS * GROUP // MINIBATCH  # minibatches per pass over a rollout: 128 stories, 16 at a time


def load_prompts(path, tokenizer, max_prompt):
    with open(path, encoding="utf-8") as f:
        docs = [parse(doc) for doc in f.read().split(EOT)]
    # TRL reads a column called prompt, and passes the others to the reward, by name.
    rows = [{"prompt": instruction(fields), "words": fields["Words"].split(", ")}
            for fields in docs if "Words" in fields and "Story" in fields]
    return Dataset.from_list([row for row in rows if len(tokenizer.encode(row["prompt"])) <= max_prompt])


def words_reward(completions, words, **kwargs):
    # Episode eight's reward: the share of the words asked for that the story uses.
    return [words_used(story, asked) / len(asked) for story, asked in zip(completions, words)]


class Report(TrainerCallback):
    # TRL runs the loop, and a callback is how our code gets a turn: a line per rollout, as grpo.py prints.
    def __init__(self, policy, tokenizer, per_rollout):
        self.policy, self.tokenizer, self.per_rollout = policy, tokenizer, per_rollout
        self.instructions = load_instructions(os.path.join(DATA_DIR, "instruct_valid.txt"), 50)
        self.start = time.time()
        self.last = None

    def evaluate(self, rollout, model):
        # TRL's weights in our model, so that grpo.py's measurement measures them.
        from_gpt2(model, self.policy)
        used = words_rate(self.policy, self.tokenizer, self.instructions)
        story = write(self.policy, self.tokenizer, DEMO, max_new_tokens=120, with_prompt=False)
        print(f"rollout {rollout:>4} | words used {used:.1%}", flush=True)
        print(textwrap.indent(textwrap.fill(story, width=86), "  "), flush=True)
        self.last = rollout

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.evaluate(0, model)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if "reward" not in logs:
            return  # the summary at the end of training
        rollout = state.global_step // self.per_rollout
        kl = f"{logs['kl']:.3f}" if "kl" in logs else "  -  "
        tied = round(logs["frac_reward_zero_std"] * PROMPTS)
        print(f"rollout {rollout:>4} | reward {logs['reward']:.3f} | kl {kl} | tied {tied:>2}/{PROMPTS} | "
              f"{(time.time() - self.start) / 60:.1f} min", flush=True)
        if rollout % EVAL_EVERY == 0:
            self.evaluate(rollout, model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.last != state.global_step // self.per_rollout:
            self.evaluate(state.global_step // self.per_rollout, model)


def main(rollouts, recipe):
    policy, tokenizer, _ = load_checkpoint(os.path.join(CHECKPOINTS, "sft", "model.pt"), pick_device())
    policy.eval()
    dataset = load_prompts(os.path.join(DATA_DIR, "instruct_train.txt"), tokenizer, policy.config.block_size - MAX_TOKENS)
    print(f"{len(dataset):,} instructions that ask for words", flush=True)
    if recipe == "matched":
        # grpo.py's choices: a KL penalty, each story's tokens averaged first, two passes over a rollout, and float32,
        # where TRL would train in bfloat16 and recompute activations for the backward pass, to save memory.
        choices = dict(beta=0.05, loss_type="grpo", num_iterations=EPOCHS, bf16=False, gradient_checkpointing=False)
    else:
        # TRL's own: no penalty, all the tokens of the batch averaged together, as DAPO does, and one pass.
        choices = dict(beta=0.0, loss_type="dapo", num_iterations=1)
    per_rollout = STEPS * choices["num_iterations"]
    run_dir = os.path.join(CHECKPOINTS, "trl", recipe)
    config = GRPOConfig(
        output_dir=run_dir,
        num_generations=GROUP,
        per_device_train_batch_size=MINIBATCH,
        steps_per_generation=STEPS,
        max_completion_length=MAX_TOKENS,
        temperature=1.0,
        epsilon=CLIP,
        learning_rate=LR,
        lr_scheduler_type="constant",
        max_steps=rollouts * per_rollout,
        logging_steps=per_rollout,
        report_to="tensorboard",
        logging_dir=os.path.join(RUNS, time.strftime(f"trl-{recipe}-%Y%m%d-%H%M%S")),
        save_strategy="no",
        dataloader_pin_memory=False,
        seed=1337,
        **choices,
    )
    report = Report(policy, tokenizer, per_rollout)
    folder = os.path.join(CHECKPOINTS, "sft-hf")
    to_gpt2(policy, tokenizer).save_pretrained(folder)
    to_hf_tokenizer(tokenizer).save_pretrained(folder)
    # A folder, as TRL loads the reference from where the model came from, and the tokenizer with it.
    trainer = GRPOTrainer(model=folder, reward_funcs=words_reward, args=config, train_dataset=dataset,
                          callbacks=[report])
    # Our line per rollout, in place of the progress bar and its dictionaries of numbers.
    trainer.remove_callback(ProgressCallback)
    trainer.train()

    from_gpt2(trainer.model, policy)
    save_checkpoint(os.path.join(run_dir, "model.pt"), policy, tokenizer, step=rollouts)
    print(f"done: {rollouts} rollouts, {(time.time() - report.start) / 60:.1f} minutes")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32, sys.argv[2] if len(sys.argv) > 2 else "matched")
