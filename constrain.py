import re

import torch

from graph_data import read_facts

# Constrained decoding: before each token is picked, the tokens that would take the text out of what's allowed get a
# logit of minus infinity, so softmax gives them no chance, and the model picks the likeliest of the rest. Here what's
# allowed is a name: inside a query's quotes, the model writes one of a list of names, and closes the quote after it.
OPENERS = (b"{name: '", b'{name: "')


class Trie:
    """Byte strings that share their beginnings: a node for each prefix, with a child for each byte that can follow."""

    def __init__(self):
        self.children = {}
        self.end = False  # a string ends here
        self.ids = []  # in the vocabulary's trie, the tokens whose bytes end here
        self.under = None  # the tokens here and under this node, worked out once

    def add(self, data, token=None):
        node = self
        for byte in data:
            node = node.children.setdefault(byte, Trie())
        node.end = True
        if token is not None:
            node.ids.append(token)

    def find(self, data):
        node = self
        for byte in data:
            node = node.children.get(byte)
            if node is None:
                return None
        return node

    def all_ids(self):
        # Each token whose bytes start with this node's.
        if self.under is None:
            self.under = self.ids + [i for child in self.children.values() for i in child.all_ids()]
        return self.under


def vocabulary(tokenizer):
    """Each token's bytes in a trie, so the tokens that start with some bytes are one walk away."""
    trie = Trie()
    for token, data in enumerate(tokenizer.token_bytes):
        trie.add(data, token)
    return trie


def allowed(vocab, name, quote):
    """The tokens whose bytes carry on from `name`, the names' node the text has got to: along a name, or to its end
    and through the closing quote, after which a token can hold anything, as "i'})" does."""
    tokens, stack = [], [(vocab, name)]
    while stack:
        v, n = stack.pop()
        # The two tries walked together, a byte at a time: a token is allowed as long as its bytes stay on a name.
        for byte, n_next in n.children.items():
            v_next = v.children.get(byte)
            if byte == quote:
                continue  # it would close the quote: a name with an apostrophe goes in double quotes
            if v_next is not None:
                tokens += v_next.ids
                stack.append((v_next, n_next))
        if n.end and quote in v.children:
            tokens += v.children[quote].all_ids()
    return tokens


class Names:
    """Keeps the names in a query to a list: given the bytes written so far, the tokens allowed next, or None for any."""

    def __init__(self, vocab, names):
        self.vocab = vocab
        self.names = Trie()
        for name in names:
            self.names.add(name.encode())

    def allowed(self, written):
        start, opener = max((written.rfind(o), o) for o in OPENERS)
        if start < 0:
            return None
        quote, name = opener[-1], written[start + len(opener):]
        if quote in name:
            return None  # the name is closed
        node = self.names.find(name)
        # Off the list already, say a name started before the constraint was on, or with no token to carry it on: then
        # the model writes it on its own.
        tokens = allowed(self.vocab, node, quote) if node is not None else None
        return tokens or None


def graph_names(tokenizer, facts):
    """The graph role's constraint: each name in its queries is a node's name, from the facts file sg serve loaded."""
    names = Names(vocabulary(tokenizer), read_facts(facts)[0])
    return lambda role, task: names if role == "graph" else None


def task_words(tokenizer):
    """The graph role's constraint: each name in its queries is copied from its task, a word or a run of words."""
    vocab = vocabulary(tokenizer)
    return lambda role, task: Names(vocab, runs(task)) if role == "graph" else None


def runs(text):
    # Each run of whole words, without the question mark or full stop after it: "the Okapi weigh?" gives "the",
    # "the Okapi", "the Okapi weigh", "Okapi", "Okapi weigh" and "weigh".
    words = [m.span() for m in re.finditer(r"\S+", text)]
    return {text[start:end].rstrip("?.,!") for i, (start, _) in enumerate(words) for _, end in words[i:]}


def written(tokenizer, ids):
    # The text so far as bytes, as the tries hold them: a token can end in the middle of a character, as é is two bytes.
    return b"".join(tokenizer.token_bytes[i] for i in ids)


def pick(logits, allowed):
    """The likeliest token, or the likeliest of the allowed ones: the others get minus infinity."""
    if allowed is not None:
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed] = 0
        logits = logits + mask
    return logits.argmax().item()


@torch.no_grad()
def main():
    import argparse

    from agent import prompt
    from checkpoint import load_checkpoint
    from model import KVCache, pick_device

    parser = argparse.ArgumentParser(description="Watch the graph role write a query, a token at a time, with its "
                                                 "names kept to a list.")
    parser.add_argument("checkpoint")
    parser.add_argument("task")
    parser.add_argument("--names", choices=["task", "graph"], default="task",
                        help="the names allowed: the task's words, or the graph's, from --facts")
    parser.add_argument("--facts", help="smollerGraph's facts file, which sg serve has loaded")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    model.eval()
    constrain = graph_names(tokenizer, args.facts) if args.names == "graph" else task_words(tokenizer)
    constraint = constrain("graph", args.task)
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt("graph", args.task))
    cache = KVCache(model.config.n_layer)
    logits, _ = model(torch.tensor([ids], device=device), cache=cache)
    out = []
    print(f"{'name so far':<22}{'most likely':<18}{'allowed':>8}{'their share':>13}   picked")
    # Up to the end of the query: after it comes sg serve's answer, which the agent writes in.
    while not written(tokenizer, out).endswith(b">>") and len(out) < 100:
        tokens = constraint.allowed(written(tokenizer, out))
        token = pick(logits[0, -1], tokens)
        if tokens is not None:
            # The likeliest token, and how much of the model's probability the allowed tokens have between them.
            probs = torch.softmax(logits[0, -1], dim=-1)
            likeliest = probs.argmax().item()
            so_far = written(tokenizer, out).rsplit(b"{name: ", 1)[1].decode(errors="replace")
            print(f"{so_far:<22}{tokenizer.id_to_token[likeliest]!r:<10}{probs[likeliest].item():>6.1%}  {len(tokens):>8,}"
                  f"{probs[tokens].sum().item():>13.1%}   {tokenizer.id_to_token[token]!r}")
        out.append(token)
        logits, _ = model(torch.tensor([[token]], device=device), cache=cache)
    print(f"\n{prompt('graph', args.task)}{tokenizer.decode(out)}")


if __name__ == "__main__":
    main()
