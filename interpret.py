import argparse

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


def main():
    from agent import prompt
    from checkpoint import load_checkpoint
    from model import pick_device

    parser = argparse.ArgumentParser(description="Look inside the model: the logit lens at the end of an answer so far.")
    parser.add_argument("checkpoint")
    parser.add_argument("task")
    parser.add_argument("answer", help="the answer so far, the role's first tokens: the lens is after its last one")
    parser.add_argument("--role", default="graph")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    model.eval()
    ids = tokenizer.encode(prompt(args.role, args.task) + args.answer)
    probs = lens(model, ids)[:, -1]
    print(f"after {tokenizer.decode(ids[-4:])!r}, the next token, layer by layer:")
    for layer, p in enumerate(probs):
        top = p.topk(3)
        guesses = "".join(f"{tokenizer.decode([i])!r:>12} {v:6.1%}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"{'embeddings' if layer == 0 else f'layer {layer}':<12}{guesses}")


if __name__ == "__main__":
    main()
