import argparse
import math
import os

import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from data import DATA_DIR
from model import pick_device
from tokenizer import EOT

CHARACTERS = 2_000_000  # of the validation stories


@torch.no_grad()
def bits_per_character(model, tokenizer, text, batch=16):
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(text))
    T = model.config.block_size
    # Back-to-back windows, each token predicted from the ones before it in its window.
    n = (len(ids) - 1) // T
    x, y = ids[: n * T].view(n, T), ids[1 : n * T + 1].view(n, T)
    nats = 0.0
    for i in range(0, n, batch):
        logits, _ = model(x[i : i + batch].to(device))
        nats += F.cross_entropy(logits.flatten(0, 1), y[i : i + batch].flatten().to(device), reduction="sum").item()
    # The windows cover nearly all the tokens; end of text counts as a single character.
    characters = len(text.replace(EOT, "\n")) * n * T / (len(ids) - 1)
    unknown = 0.0 if tokenizer.unk_id is None else (ids == tokenizer.unk_id).float().mean().item()
    # The stories that come back from encoding and decoding as they were, spaces and new lines included.
    stories = [story.strip() for story in text.split(EOT) if story.strip()]
    round_trip = sum(tokenizer.decode(tokenizer.encode(story)) == story for story in stories) / len(stories)
    return {"bits_per_character": nats / math.log(2) / characters, "tokens": len(ids), "unknown": unknown,
            "characters_per_token": len(text.replace(EOT, "\n")) / len(ids), "round_trip": round_trip}


def main():
    parser = argparse.ArgumentParser(description="Bits per character on held-out stories, for models with any tokenizer.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    with open(os.path.join(DATA_DIR, "valid.txt"), encoding="utf-8") as f:
        text = f.read(CHARACTERS)
    print(f"{'model':<20}{'tokens':>10}{'unknown':>9}{'chars/token':>13}{'round trip':>12}{'bits/char':>11}")
    for path in args.checkpoints:
        model, tokenizer, _ = load_checkpoint(path, args.device)
        model.eval()
        m = bits_per_character(model, tokenizer, text)
        name = os.path.basename(os.path.dirname(path))
        print(f"{name:<20}{m['tokens']:>10,}{m['unknown']:>9.2%}{m['characters_per_token']:>13.2f}"
              f"{m['round_trip']:>12.1%}{m['bits_per_character']:>11.3f}")


if __name__ == "__main__":
    main()
