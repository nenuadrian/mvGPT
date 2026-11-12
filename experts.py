import argparse
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch

from checkpoint import load_checkpoint
from data import DATA_DIR
from model import pick_device
from tokenizer import EOT, SPECIALS

# A colour for each expert, as 24-bit terminal colours that read well on a dark background.
COLOURS = [(239, 83, 80), (255, 167, 38), (255, 238, 88), (102, 187, 106),
           (38, 198, 218), (66, 165, 245), (171, 71, 188), (236, 64, 122)]


def paint(text, expert):
    r, g, b = COLOURS[expert % len(COLOURS)]
    return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"


@torch.no_grad()
def first_choices(model, tokens, device, batch=48):
    # Each token's first choice of expert, in each block: an array of blocks by tokens.
    window = model.config.block_size
    windows = torch.from_numpy(tokens[: len(tokens) // window * window].astype(np.int64)).view(-1, window)
    choices = []
    for i in range(0, len(windows), batch):
        model(windows[i : i + batch].to(device))
        choices.append(torch.stack([b.mlp.chosen[:, 0] for b in model.blocks]).cpu())
    return torch.cat(choices, dim=1).numpy()


def spaced(words):
    # Each word with the space that goes before it, as the tokenizer's detokenize would put it.
    out = []
    for i, w in enumerate(words):
        glued = i == 0 or re.match(r"[.,!?;:]", w) or w == "'" or words[i - 1] == "'"
        out.append(("" if glued else " ") + w)
    return out


def main():
    parser = argparse.ArgumentParser(description="Which words go to which expert, in a mixture of experts.")
    parser.add_argument("checkpoint")
    parser.add_argument("--layer", type=int, help="the block to look at (default: the middle one)")
    parser.add_argument("--tokens", type=int, default=100_000, help="held-out tokens to route")
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    n_expert = model.config.n_expert
    layer = model.config.n_layer // 2 if args.layer is None else args.layer
    tokens = np.memmap(os.path.join(DATA_DIR, "valid.bin"), dtype=np.uint16, mode="r")[: args.tokens]
    choices = first_choices(model, tokens, device)
    tokens = tokens[: choices.shape[1]]

    # Each expert's share of the first choices, block by block: 1 / n_expert each, when they're even.
    print("first choices, by block:   " + " ".join(paint(f"  e{e}", e) for e in range(n_expert)))
    for b, row in enumerate(choices):
        shares = np.bincount(row, minlength=n_expert) / len(row)
        print(f"  block {b}:                 " + " ".join(f"{s:4.0%}" for s in shares))

    # For each expert in the block, the words it gets most often, each with the share of its uses that went there.
    seen = Counter(tokens.tolist())
    routed = defaultdict(Counter)
    for token, expert in zip(tokens.tolist(), choices[layer].tolist()):
        routed[expert][token] += 1
    print(f"\nblock {layer}, the words each expert gets most often, with the share of their uses:")
    for e in range(n_expert):
        top = [(n, t) for t, n in routed[e].most_common() if tokenizer.id_to_token[t] not in SPECIALS][:8]
        words = ", ".join(f"{tokenizer.id_to_token[t]} {n / seen[t]:.0%}" for n, t in top)
        print(paint(f"  e{e}: ", e) + (words or "no tokens"))

    # The first whole held-out story, after the first end of text, each word in the colour of its expert there.
    ids = tokens.tolist()
    eot = tokenizer.encode(EOT)[0]
    start = ids.index(eot) + 1
    end = min(ids.index(eot, start), start + 80)
    words = [tokenizer.id_to_token[t] for t in ids[start:end]]
    painted = [paint(w, e) for w, e in zip(spaced(words), choices[layer][start:end].tolist())]
    print(f"\na held-out story, each word in its expert's colour in block {layer}:\n" + "".join(painted))


if __name__ == "__main__":
    main()
