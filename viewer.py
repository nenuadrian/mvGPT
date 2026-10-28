import argparse
import base64
import json
import os

from interpret import attention, lens

# A page to look at attention with, viewer.html, written for a text: its tokens, shaded by how much the token picked
# takes from each, for a head or a whole layer; each head's pattern, small, in a grid; and the logit lens at the token
# picked. The numbers go in the page, in place of its marker, so a browser opens it with nothing else running.
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")


def main():
    from agent import prompt
    from checkpoint import load_checkpoint
    from model import pick_device

    parser = argparse.ArgumentParser(description="Write a page to look at the model's attention on a role's prompt.")
    parser.add_argument("checkpoint")
    parser.add_argument("task")
    parser.add_argument("answer", help="the answer's text, which goes in after the prompt")
    parser.add_argument("--role", default="graph")
    parser.add_argument("--out", default="attention.html")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    model.eval()
    ids = tokenizer.encode(prompt(args.role, args.task) + args.answer)
    weights = attention(model, ids)
    probs = lens(model, ids)
    top = probs.max(dim=-1)
    data = {
        "what": f"{args.role}: {args.task}",
        "tokens": [tokenizer.decode([i]) for i in ids],
        "layers": weights.size(0), "heads": weights.size(1),
        # Each weight as a byte, 0 to 255, so the page holds 144 heads of a prompt in under a megabyte.
        "weights": base64.b64encode((weights.clamp(0, 1) * 255).round().byte().cpu().numpy().tobytes()).decode(),
        "lens": [[[tokenizer.decode([top.indices[l, t].item()]), round(top.values[l, t].item(), 3)]
                  for l in range(probs.size(0))] for t in range(len(ids))],
        "pick": len(ids) - 1, "layer": 0, "head": 0,
    }
    with open(PAGE) as f:
        page = f.read()
    with open(args.out, "w") as f:
        f.write(page.replace("__DATA__", json.dumps(data)))
    print(f"wrote {args.out}: {len(ids)} tokens, {data['layers']} layers of {data['heads']} heads")


if __name__ == "__main__":
    main()
