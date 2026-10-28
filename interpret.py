import argparse
import json
import os
import random
import re

import torch

# Looking inside the model: each layer's attention weights on a text, the logit lens on its residual stream, and heads
# turned off to see what they were doing.


@torch.no_grad()
def attention(model, ids):
    """The attention weights of each layer's heads on ids: layers, heads, tokens, and the tokens each looks back at."""
    device = next(model.parameters()).device
    for block in model.blocks:
        block.attn.keep = True
    try:
        model(torch.tensor([ids], device=device))
    finally:
        for block in model.blocks:
            block.attn.keep = False
    return torch.stack([block.attn.weights[0] for block in model.blocks])


@torch.no_grad()
def lens(model, ids):
    """The logit lens: the final norm and the head on the residual stream after the embeddings and after each block,
    so each layer's guess at the next token shows. Probabilities: layers + 1, tokens, vocabulary."""
    device = next(model.parameters()).device
    streams = []
    # A hook gets each block's output, the residual stream so far, as the model runs.
    hooks = [block.register_forward_hook(lambda module, inputs, out: streams.append(out)) for block in model.blocks]
    x = torch.tensor([ids], device=device)
    try:
        model(x)
    finally:
        for hook in hooks:
            hook.remove()
    streams = [model.embed_tokens(x)] + streams
    return torch.stack([torch.softmax(model.lm_head(model.norm(s))[0], dim=-1) for s in streams])


class HeadsOff:
    """Heads turned off while it's open, each a (layer, head) from 0: with HeadsOff(model, [(11, 0)]): ..."""

    def __init__(self, model, heads):
        self.model, self.heads = model, heads

    def __enter__(self):
        for layer, head in self.heads:
            self.model.blocks[layer].attn.off.append(head)

    def __exit__(self, *exc):
        for block in self.model.blocks:
            block.attn.off = []


def span(tokenizer, ids, text, after=0):
    """The positions of the tokens that hold `text`, the first time it's written after character `after`."""
    pieces = [tokenizer.decode([i]) for i in ids]
    ends = [sum(len(p) for p in pieces[: i + 1]) for i in range(len(pieces))]
    start = "".join(pieces).index(text, after)
    return [i for i, end in enumerate(ends) if end > start and end - len(pieces[i]) < start + len(text)]


@torch.no_grad()
def copy_scores(model, tokenizer, examples):
    """For each head, the share of its attention that goes to the name in the task, from the tokens that write the name
    into the query: its opening quote, and each of its tokens but the last. Averaged over the examples: layers, heads.
    """
    total, count = 0, 0
    for e in examples:
        ids = tokenizer.encode(e["segments"][0][0] + e["segments"][1][0])
        text = tokenizer.decode(ids)
        name = NAME.search(text)
        try:
            in_task = span(tokenizer, ids, name[2], text.index("Task: "))
            in_query = span(tokenizer, ids, name[1] + name[2], name.start())
        except ValueError:
            continue  # a token that ends inside a letter, as é is two bytes: the name isn't in the tokens' text
        weights = attention(model, ids)
        total, count = total + weights[:, :, in_query[:-1]][..., in_task].sum(-1).mean(-1), count + 1
    return total / count


# A name in a query, in single quotes or in double ones, with the quote it opened with.
NAME = re.compile(r"\{name: (['\"])(.+?)\1\}")


def names_right(examples, rows):
    """Of the answers whose query names something, the share that named what the example's query does: None when no
    query names anything."""
    named = [(NAME.search(e["segments"][1][0])[2], NAME.search(r["trace"])) for e, r in zip(examples, rows)]
    named = [(want, got[2]) for want, got in named if got is not None]
    return sum(want == got for want, got in named) / len(named) if named else None


def share(value):
    return "-" if value is None else f"{value:.0%}"


def heads(model, tokenizer, server, limit):
    """Which heads look at the name in the task as the query copies it, and the graph role's questions with them
    turned off."""
    import graph_data
    from agent import Agent
    from agent_eval import attempt
    from graph_agent_data import as_role

    # Episode 24's held-out questions in the training's wordings, under the graph role's line, as episode 25 asked them.
    examples = [as_role(e) for e in graph_data.load("seen")[:limit]]
    with_names = [e for e in examples if NAME.search(e["segments"][1][0])]
    scores = copy_scores(model, tokenizer, with_names)
    L, H = scores.shape
    order = sorted(((l, h) for l in range(L) for h in range(H)), key=lambda lh: -scores[lh].item())
    print(f"the heads that look at the name in the task, of {L * H}, over {len(with_names)} questions:")
    print("  ".join(f"layer {l + 1} head {h + 1}: {scores[l, h]:.0%}" for l, h in order[:6]))

    def right(off):
        with HeadsOff(model, off):
            rows = [attempt(lambda sandbox: Agent(model, tokenizer, sandbox, server=server), e) for e in examples]
        return sum(r["right"] for r in rows) / len(rows), rows

    results = {"scores": scores.tolist(), "top": [[l + 1, h + 1] for l, h in order[:8]], "off": {}}
    none, before = right([])
    print(f"\nthe graph role, {len(examples)} questions: right, and the names copied right\n"
          f"{'heads off':<12}{'the top ones':>14}{'names':>9}{'random ones':>14}{'names':>9}")
    print(f"{0:<12}{none:>14.1%}{share(names_right(examples, before)):>9}", flush=True)
    results["names"] = names_right(examples, before)
    for k in (1, 2, 4, 8):
        top, rows = right(order[:k])
        # As many heads again, three times over, from all 144: what turning off any k heads does.
        draws = [right(random.Random(seed).sample(order, k)) for seed in range(3)]
        rand = sum(d[0] for d in draws) / 3
        rand_names = [names_right(examples, d[1]) for d in draws]
        rand_names = None if None in rand_names else sum(rand_names) / 3
        results["off"][k] = {"top": top, "top_names": names_right(examples, rows), "random": [d[0] for d in draws],
                             "random_names": rand_names}
        print(f"{k:<12}{top:>14.1%}{share(names_right(examples, rows)):>9}{rand:>14.1%}{share(rand_names):>9}",
              flush=True)
    # A question the role got right, and wrong with the top eight off.
    wrong = next(r["trace"] for r, b in zip(rows, before) if b["right"] and not r["right"])
    print(f"\nwith the top 8 off:\n{wrong}")
    results["right"] = none
    return results


def main():
    from agent import prompt
    from checkpoint import load_checkpoint
    from graph import SERVER
    from hellaswag import RESULTS
    from model import pick_device

    parser = argparse.ArgumentParser(description="Look inside the model: the logit lens, and heads turned off.")
    parser.add_argument("checkpoint")
    commands = parser.add_subparsers(dest="command", required=True)
    at = commands.add_parser("lens", help="the logit lens at the end of a role's answer so far")
    at.add_argument("task")
    at.add_argument("answer", help="the answer so far: the lens is after its last token")
    at.add_argument("--role", default="graph")
    copying = commands.add_parser("heads", help="the heads that look at the name to copy, and the role without them")
    copying.add_argument("--limit", type=int, default=200, help="the questions to ask")
    copying.add_argument("--server", default=SERVER, help="where sg serve is listening")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    model.eval()
    if args.command == "heads":
        results = heads(model, tokenizer, args.server, args.limit)
        name = os.path.basename(os.path.dirname(args.checkpoint))
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, f"heads-{name}.json"), "w") as f:
            json.dump(results, f, indent=1)
        return
    ids = tokenizer.encode(prompt(args.role, args.task) + args.answer)
    probs = lens(model, ids)[:, -1]
    print(f"after {tokenizer.decode(ids[-4:])!r}, the next token, layer by layer:")
    for layer, p in enumerate(probs):
        top = p.topk(3)
        pairs = zip(top.values.tolist(), top.indices.tolist())
        guesses = "".join(f"{tokenizer.decode([i])!r:>12} {v:6.1%}" for v, i in pairs)
        print(f"{'embeddings' if layer == 0 else f'layer {layer}':<12}{guesses}")


if __name__ == "__main__":
    main()
