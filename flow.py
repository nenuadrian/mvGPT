import argparse
import json
import os

import torch

from checkpoint import load_checkpoint
from hellaswag import RESULTS
from interpret import attention
from model import pick_device

# How the numbers flow for one text, in a block of each kind. Attention: a head's weights, from comparing each token's
# query with the keys before it. Hyena: a channel's gates and filter, the matrix they make, and the last token's
# output worked through term by term.


@torch.no_grad()
def hyena_parts(model, ids, layer):
    # What block `layer`'s Hyena sees, from a hook on the norm before it, and what it makes of it, channel by channel.
    block = model.blocks[layer]
    seen = []
    hook = block.attn_norm.register_forward_hook(lambda module, inputs, out: seen.append(out))
    try:
        model(torch.tensor([ids], device=next(model.parameters()).device))
    finally:
        hook.remove()
    hyena = block.hyena
    x1, x2, v = (s[0] for s in hyena.streams(seen[0]))
    return {"x1": x1, "x2": x2, "v": v, "h": hyena.filter(len(ids)), "b": hyena.filter.bias,
            "W": hyena.matrices(seen[0]), "filters": hyena.filter(hyena.filter.length)}


@torch.no_grad()
def next_tokens(model, tokenizer, ids, k=3):
    # The model's k likeliest next tokens after the text, with their probabilities.
    logits, _ = model(torch.tensor([ids], device=next(model.parameters()).device))
    top = logits[0, -1].softmax(dim=-1).topk(k)
    return ", ".join(f"{tokenizer.decode([i])!r} {p:.1%}" for p, i in zip(top.values.tolist(), top.indices.tolist()))


def far_channel(W):
    # The channel whose last token draws most on tokens more than two back, by the size of their weights.
    last = W[:, -1, :].abs()
    return (last[:, :-3].sum(-1) / last.sum(-1)).argmax().item()


def main():
    parser = argparse.ArgumentParser(description="One text through an attention model and a Hyena model: a head's "
                                     "weights, and a channel's gates, filter and matrix, with its last output worked out.")
    parser.add_argument("attention", help="an attention model's checkpoint")
    parser.add_argument("hyena", help="a Hyena model's checkpoint")
    parser.add_argument("text")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--channel", type=int, help="by default, the one whose last token looks furthest back")
    parser.add_argument("--rows", type=int, default=6, help="terms of the last output to print")
    parser.add_argument("--out", default=os.path.join(RESULTS, "flow.json"))
    args = parser.parse_args()
    device = pick_device()

    model, tokenizer, _ = load_checkpoint(args.attention, device)
    model.eval()
    ids = tokenizer.encode(args.text)
    tokens = [tokenizer.decode([i]) for i in ids]
    t = len(ids) - 1
    A = attention(model, ids)[args.layer, args.head]
    print(f"{len(ids)} tokens. Attention, block {args.layer}, head {args.head}: what {tokens[t]!r} takes from each token")
    for s in A[t].argsort(descending=True)[: args.rows].tolist():
        print(f"  {s:>3} {tokens[s]!r:<14}{A[t, s].item():>8.3f}")
    print(f"  next: {next_tokens(model, tokenizer, ids)}")

    model, _, _ = load_checkpoint(args.hyena, device)
    model.eval()
    p = hyena_parts(model, ids, args.layer)
    c = far_channel(p["W"]) if args.channel is None else args.channel
    x1, x2, v, h, b = p["x1"][c], p["x2"][c], p["v"][c], p["h"][c], p["b"][c].item()
    z = x1 * v
    print(f"\nHyena, block {args.layer}, channel {c}: y_{t} = x2_{t} * (sum of h_({t}-s) * z_s + b * z_{t}), z = x1 * v")
    print(f"  {'s':>3} {'token':<14}{'x1':>8}{'v':>8}{'z':>8}{'h':>8}{'h * z':>9}")
    terms = h[: t + 1].flip(0) * z
    for s in range(t, max(-1, t - args.rows), -1):
        print(f"  {s:>3} {tokens[s]!r:<14}{x1[s].item():>8.3f}{v[s].item():>8.3f}{z[s].item():>8.3f}"
              f"{h[t - s].item():>8.3f}{terms[s].item():>9.3f}")
    total = terms.sum().item()
    rest = terms[: max(0, t + 1 - args.rows)].sum().item()
    y = x2[t].item() * (total + b * z[t].item())
    print(f"  the {max(0, t + 1 - args.rows)} tokens before them {rest:.3f}, sum over all {t + 1} tokens {total:.3f}")
    print(f"  skip b * z = {b:.3f} * {z[t].item():.3f}, x2 = {x2[t].item():.3f}: y = {y:.3f}")
    print(f"  next: {next_tokens(model, tokenizer, ids)}")
    # The same output, as the channel's matrix times its values: row t of D(x2) (S(h) + b I) D(x1), times v.
    assert abs(y - (p["W"][c, t] @ v).item()) < 1e-3 * max(1, abs(y))

    out = {"tokens": tokens, "layer": args.layer, "head": args.head, "channel": c,
           "attention": A.tolist(), "hyena": p["W"][c].tolist(),
           "x1": x1.tolist(), "x2": x2.tolist(), "v": v.tolist(), "h": h.tolist(), "b": b, "y": y,
           # Four channels' whole filters, from the slowest to fade to the fastest, and the one picked.
           "filters": {str(k): p["filters"][k].tolist()
                       for k in sorted({0, len(p["b"]) // 3, 2 * len(p["b"]) // 3, len(p["b"]) - 1, c})}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
