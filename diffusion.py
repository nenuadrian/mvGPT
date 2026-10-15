import argparse
import os
import textwrap

import torch
from torch.nn import functional as F

from data import DATA_DIR, load_tokenizer
from tokenizer import EOT


def mask_id(tokenizer):
    # Diffusion's one new token, [MASK], goes after the tokenizer's: the last row of the model's
    # embedding. It's never in the stories, only where the noise hides a token.
    return len(tokenizer)


def add_noise(x, rates, mask):
    # Each token becomes [MASK] with its row's probability, except the first, so there's always one to
    # start from.
    hidden = torch.rand(x.shape, device=x.device) < rates[:, None]
    hidden[:, 0] = False
    return x.masked_fill(hidden, mask), hidden


def hidden_losses(model, x, rates, mask):
    noisy, hidden = add_noise(x, rates, mask)
    logits, _ = model(noisy)
    # The head still predicts each position's next token: a [MASK] at position i is filled from the
    # output at position i - 1, which now sees the whole sequence, before it and after.
    targets = x[:, 1:].masked_fill(~hidden[:, 1:], -1)
    nats = F.cross_entropy(logits[:, :-1].flatten(0, 1), targets.flatten(), ignore_index=-1, reduction="none")
    # The loss on each hidden token, and 0 for the ones it could see.
    return nats.view(x.size(0), -1), hidden[:, 1:]


def diffusion_loss(model, x, mask):
    B, T = x.shape
    # A mask rate per row, spread evenly from 0 to 1 across the batch from one random offset, so every
    # batch has rows with a few tokens hidden and rows with nearly all of them.
    rates = ((torch.rand(1, device=x.device) + torch.arange(B, device=x.device) / B) % 1).clamp(min=1e-3)
    nats, _ = hidden_losses(model, x, rates, mask)
    # Each row's loss on its hidden tokens, divided by its rate: averaged over the rates, it's a bound on
    # the negative log likelihood. Then per token, like next-token prediction's loss.
    return (nats.sum(1) / (rates * (T - 1))).mean()


def show(tokenizer, ids, mask):
    # The text, with a shaded block for each token that's still hidden.
    text, run = "", []
    for i in ids:
        if i == mask:
            text += tokenizer.decode(run) + " ░"
            run = []
        else:
            run.append(i)
    return (text + tokenizer.decode(run)).lstrip()


def masking(args):
    tokenizer = load_tokenizer(bpe=True)
    with open(os.path.join(DATA_DIR, "valid.txt"), encoding="utf-8") as f:
        stories = [story.strip() for story in f.read(100_000).split(EOT)][1:]
    # A short held-out story, after end of text, as the windows in training have it.
    story = next(story for story in stories if len(story) < 600)
    x = torch.tensor([[tokenizer.eot_id] + tokenizer.encode(story)])
    mask = mask_id(tokenizer)
    for rate in args.rates:
        noisy, hidden = add_noise(x, torch.tensor([rate]), mask)
        print(f"{rate:.0%}: {hidden.sum().item()} of {x.size(1) - 1} tokens hidden")
        print(textwrap.fill(show(tokenizer, noisy[0].tolist(), mask), width=90), end="\n\n")


def main():
    parser = argparse.ArgumentParser(description="Masked diffusion: hide a story's tokens, and write stories by unhiding them.")
    commands = parser.add_subparsers(required=True)
    mask = commands.add_parser("mask", help="a held-out story with some of its tokens hidden")
    mask.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    mask.set_defaults(run=masking)
    for command in commands.choices.values():
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.run(args)


if __name__ == "__main__":
    main()
