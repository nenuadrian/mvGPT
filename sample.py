"""Writes stories with a trained model.

    python sample.py checkpoints/pretrain/model.pt "Once upon a time"
"""
import argparse
import textwrap

import torch

from checkpoint import load_checkpoint
from model import pick_device


def write(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50, with_prompt=True):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens, temperature, top_k, eos_id=tokenizer.eot_id)
    model.train(was_training)
    return tokenizer.decode(out[0].tolist() if with_prompt else out[0, ids.size(1) :].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("prompt", nargs="?", default="Once upon a time")
    parser.add_argument("-n", "--stories", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    for _ in range(args.stories):
        story = write(model, tokenizer, args.prompt, args.tokens, args.temperature, args.top_k)
        print(textwrap.fill(story, width=90), end="\n\n")


if __name__ == "__main__":
    main()
