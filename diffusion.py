import argparse
import os
import textwrap

import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from data import DATA_DIR, load_tokenizer
from model import pick_device
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


def from_autoregressive(model, path):
    # A diffusion model that starts from a next-token model's weights, as Dream 7B started from Qwen's:
    # the same weights everywhere, with the new model's row for [MASK] added to the embedding.
    state = torch.load(path, map_location="cpu")["model"]
    own = model.state_dict()
    for name in ("embed_tokens.weight", "lm_head.weight"):
        state[name] = torch.cat([state[name], own[name][-1:].cpu()])
    model.load_state_dict(state)


@torch.no_grad()
def denoise(model, canvas, mask, steps, temperature=1.0, top_k=None, ladder=None):
    # Writes each [MASK] in the canvas, some of them at each step, and leaves the other tokens as they are.
    canvas = canvas.clone()
    for step in range(steps):
        hidden = canvas == mask
        left = hidden.sum(1)
        if left.sum() == 0:
            break
        logits, _ = model(canvas)
        # Position i - 1's output predicts token i, so the first token can't be hidden. In float32, as
        # on a GPU they can come in bfloat16.
        logits = logits[:, :-1].float() / temperature
        logits[..., mask] = -float("inf")  # the answer is never [MASK]
        if top_k is not None:
            kth = torch.topk(logits, top_k).values[..., [-1]]
            logits[logits < kth] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        # A guess for every position at once, and how likely the model found it.
        guess = torch.multinomial(probs.flatten(0, 1), 1).view(probs.shape[:2])
        sure = probs.gather(-1, guess[..., None])[..., 0]
        guess, sure = F.pad(guess, (1, 0)), F.pad(sure, (1, 0))
        sure[~hidden] = -1.0
        # Of the hidden tokens, the surest guesses are kept: enough of them that the last step writes the
        # rest. The others stay hidden, for a fresh guess next step, with more of the story around them.
        keep = torch.ceil(left / (steps - step))
        rank = sure.argsort(dim=1, descending=True).argsort(dim=1)
        reveal = hidden & (rank < keep[:, None])
        canvas[reveal] = guess[reveal]
        if ladder is not None:
            ladder.append(canvas[0].tolist())
    return canvas


def canvas_for(model, tokenizer, ids):
    # A window of [MASK], the length the model trained on, with ids at its start.
    device = next(model.parameters()).device
    canvas = torch.full((1, model.config.block_size), mask_id(tokenizer), device=device)
    canvas[0, : len(ids)] = torch.tensor(ids, device=device)
    return canvas


def first_story(tokenizer, ids, tokens=None):
    # The canvas starts with end of text, as a story does in training, after the one before it. Its story
    # runs to the next end of text, and what's after that is the start of another story.
    end = ids.index(tokenizer.eot_id, 1) if tokenizer.eot_id in ids[1:] else len(ids)
    if tokens is not None:
        end = min(end, 1 + tokens)
    return tokenizer.decode(ids[1:end])


def write(model, tokenizer, prompt, steps=64, temperature=0.8, top_k=50, tokens=None, ladder=None):
    # A story by denoising: end of text and the prompt, then [MASK] to the end of the window.
    was_training = model.training
    model.eval()
    canvas = canvas_for(model, tokenizer, [tokenizer.eot_id] + tokenizer.encode(prompt))
    out = denoise(model, canvas, mask_id(tokenizer), steps, temperature, top_k, ladder)
    model.train(was_training)
    return first_story(tokenizer, out[0].tolist(), tokens)


def fill(model, tokenizer, before, after, gap, steps=64, temperature=0.8, top_k=50):
    # Infilling: a story's beginning and its end, with `gap` tokens of [MASK] between them, and after its
    # end, the rest of the window, as in training.
    model.eval()
    mask = mask_id(tokenizer)
    ids = [tokenizer.eot_id] + tokenizer.encode(before) + [mask] * gap + tokenizer.encode(" " + after.strip())
    out = denoise(model, canvas_for(model, tokenizer, ids + [tokenizer.eot_id]), mask, steps, temperature, top_k)
    return first_story(tokenizer, out[0].tolist())


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


def writing(args):
    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    mask = mask_id(tokenizer)
    for _ in range(args.stories):
        ladder = []
        story = write(model, tokenizer, args.prompt, args.steps, args.temperature, args.top_k, ladder=ladder)
        if args.ladder:
            # The canvas along the way, as far as its first story goes once it's written.
            final = ladder[-1]
            end = final.index(tokenizer.eot_id, 1) if tokenizer.eot_id in final[1:] else len(final)
            for step in args.ladder:
                print(f"step {step}:")
                print(textwrap.fill(show(tokenizer, ladder[step - 1][1:end], mask), width=90), end="\n\n")
        else:
            print(textwrap.fill(story, width=90), end="\n\n")


def filling(args):
    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    for _ in range(args.stories):
        story = fill(model, tokenizer, args.before, args.after, args.gap, args.steps, args.temperature, args.top_k)
        print(textwrap.fill(story, width=90), end="\n\n")


def main():
    parser = argparse.ArgumentParser(description="Masked diffusion: hide a story's tokens, and write stories by unhiding them.")
    commands = parser.add_subparsers(required=True)
    mask = commands.add_parser("mask", help="a held-out story with some of its tokens hidden")
    mask.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    mask.set_defaults(run=masking)
    for name, run in (("write", writing), ("fill", filling)):
        command = commands.add_parser(name)
        command.add_argument("checkpoint")
        command.add_argument("-n", "--stories", type=int, default=1)
        command.add_argument("--steps", type=int, default=64, help="passes of the model, each unhiding some tokens")
        command.add_argument("--temperature", type=float, default=0.8)
        command.add_argument("--top-k", type=int, default=50)
        command.set_defaults(run=run)
    commands.choices["write"].add_argument("prompt", nargs="?", default="Once upon a time")
    commands.choices["write"].add_argument("--ladder", type=int, nargs="+", help="show the canvas after these steps")
    commands.choices["fill"].add_argument("before")
    commands.choices["fill"].add_argument("after")
    commands.choices["fill"].add_argument("--gap", type=int, default=40, help="tokens to write between them")
    for command in commands.choices.values():
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.run(args)


if __name__ == "__main__":
    main()
