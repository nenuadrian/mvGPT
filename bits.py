import argparse
import json
import math
import os

import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from data import DATA_DIR
from diffusion import hidden_losses, mask_id
from model import pick_device
from tokenizer import EOT

CHARACTERS = 2_000_000  # of the validation stories
RATES = 16  # slices of 0 to 1, for a diffusion model's mask rates


@torch.no_grad()
def bits_per_character(model, tokenizer, text, batch=16):
    device = next(model.parameters()).device
    ids = torch.tensor(tokenizer.encode(text))
    T = model.config.block_size
    # Back-to-back windows, each token predicted from the ones before it in its window.
    n = (len(ids) - 1) // T
    x, y = ids[: n * T].view(n, T), ids[1 : n * T + 1].view(n, T)
    nats, predicted = 0.0, 0
    rate_nats, rate_hidden = [0.0] * RATES, [0] * RATES
    torch.manual_seed(0)
    for i in range(0, n, batch):
        xb = x[i : i + batch].to(device)
        if model.config.causal:
            logits, _ = model(xb)
            nats += F.cross_entropy(logits.flatten(0, 1), y[i : i + batch].flatten().to(device), reduction="sum").item()
            predicted += xb.numel()
            continue
        # A diffusion model's loss is a bound: its loss on the hidden tokens at a mask rate from each of 16
        # equal slices of 0 to 1, divided by the rate, averaged over the slices. The first token is given.
        for k in range(RATES):
            rates = (k + torch.rand(len(xb), device=device)) / RATES
            losses, hidden = hidden_losses(model, xb, rates, mask_id(tokenizer))
            nats += (losses.sum(1) / rates).sum().item() / RATES
            rate_nats[k] += losses.sum().item()
            rate_hidden[k] += hidden.sum().item()
        predicted += xb[:, 1:].numel()
    # The windows cover nearly all the tokens; end of text counts as a single character.
    characters = len(text.replace(EOT, "\n")) * predicted / (len(ids) - 1)
    unknown = 0.0 if tokenizer.unk_id is None else (ids == tokenizer.unk_id).float().mean().item()
    # The stories that come back from encoding and decoding as they were, spaces and new lines included.
    stories = [story.strip() for story in text.split(EOT) if story.strip()]
    round_trip = sum(tokenizer.decode(tokenizer.encode(story)) == story for story in stories) / len(stories)
    results = {"bits_per_character": nats / math.log(2) / characters, "loss": nats / predicted, "tokens": len(ids),
               "unknown": unknown, "characters_per_token": len(text.replace(EOT, "\n")) / len(ids),
               "round_trip": round_trip}
    if not model.config.causal:
        # The loss per hidden token at each rate, from a few tokens hidden to nearly all of them.
        results["by_rate"] = [{"rate": (k + 0.5) / RATES, "loss": rate_nats[k] / rate_hidden[k]} for k in range(RATES)]
    return results


def main():
    parser = argparse.ArgumentParser(description="Bits per character on held-out stories, for models with any tokenizer.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--out", help="a file to write the measurements to, as JSON")
    args = parser.parse_args()

    with open(os.path.join(DATA_DIR, "valid.txt"), encoding="utf-8") as f:
        text = f.read(CHARACTERS)
    print(f"{'model':<40}{'tokens':>10}{'unknown':>9}{'chars/token':>13}{'round trip':>12}{'bits/char':>11}")
    rows = []
    for path in args.checkpoints:
        model, tokenizer, _ = load_checkpoint(path, args.device)
        model.eval()
        m = bits_per_character(model, tokenizer, text)
        name = os.path.basename(os.path.dirname(path))
        rows.append({"model": name, **m})
        # A diffusion model's is at most this: it's a bound.
        bound = "<=" if "by_rate" in m else ""
        print(f"{name:<40}{m['tokens']:>10,}{m['unknown']:>9.2%}{m['characters_per_token']:>13.2f}"
              f"{m['round_trip']:>12.1%}{bound:>5}{m['bits_per_character']:>6.3f}")
    for row in rows:
        if "by_rate" in row:
            print(f"\n{row['model']}, loss per hidden token, by the share of tokens hidden:")
            for r in row["by_rate"]:
                print(f"  {r['rate']:>4.0%} {r['loss']:6.2f} " + "#" * round(r["loss"] * 8))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
