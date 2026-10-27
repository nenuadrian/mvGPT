import torch

from model import KVCache

# Speculative decoding: a cheap draft guesses the next few tokens, and the model checks them all in one forward pass.
# At batch 1 on a GPU, a pass costs about the same for a few tokens as for one, as with a model this small, most of its
# time goes into launching its operations. The model keeps the guesses up to the first it disagrees with, and its own
# token there, so the text is the one greedy decoding writes, in fewer passes.


class Lookup:
    """Guesses from the text so far: what came after the last time its last few tokens were written. An answer copies
    names and numbers from its task and from what the tools gave back, and those it guesses for nothing."""

    def __init__(self, k=4, longest=3):
        self.k, self.longest = k, longest  # how many tokens to guess, and the longest tail to look for

    def guess(self, tokens):
        for n in range(self.longest, 0, -1):
            tail = tokens[-n:]
            # The latest earlier place the tail was written, and the tokens after it there.
            for start in range(len(tokens) - n - 1, -1, -1):
                if tokens[start:start + n] == tail:
                    return tokens[start + n:start + n + self.k]
        return []


class Draft:
    """Guesses with a small model of the same tokenizer, greedily, with a KV cache of its own."""

    def __init__(self, model, k=4):
        self.model, self.k = model, k
        self.cache, self.fed = KVCache(model.config.n_layer), []

    @torch.no_grad()
    def guess(self, tokens):
        device = next(self.model.parameters()).device
        # The cache keeps the tokens it was fed that are still the text's, and the rest of the text goes in after them.
        same = 0
        while same < min(len(self.fed), len(tokens)) and self.fed[same] == tokens[same]:
            same += 1
        same = min(same, len(tokens) - 1)  # the last token goes in again, for the logits after it
        self.cache.trim(len(self.fed) - same)
        logits, _ = self.model(torch.tensor([tokens[same:]], device=device), cache=self.cache)
        self.fed = list(tokens)
        out = []
        for i in range(self.k):
            out.append(logits[0, -1].argmax().item())
            if i < self.k - 1:
                logits, _ = self.model(torch.tensor([out[-1:]], device=device), cache=self.cache)
                self.fed.append(out[-1])
        return out


@torch.no_grad()
def passes(model, ids, draft=None):
    """Greedy decoding after ids, a pass at a time. Each pass yields the tokens it wrote, the model's own and the
    guesses it kept, and how many guesses it was given. The caller stops when it has what it needs."""
    device = next(model.parameters()).device
    cache = KVCache(model.config.n_layer)
    logits, _ = model(torch.tensor([ids], device=device), cache=cache)
    after, out = logits[0, -1], []
    while len(ids) + len(out) < model.config.block_size:
        # The model's own next token, from the pass before, and the draft's guesses at the tokens after it.
        token = after.argmax().item()
        room = model.config.block_size - len(ids) - len(out) - 1
        guesses = draft.guess(ids + out + [token])[:room] if draft else []
        logits, _ = model(torch.tensor([[token] + guesses], device=device), cache=cache)
        # The token the model writes after each of them: a guess is kept while it's that token.
        wrote = logits[0].argmax(dim=-1).tolist()
        n = 0
        while n < len(guesses) and guesses[n] == wrote[n]:
            n += 1
        cache.trim(len(guesses) - n)
        after = logits[0, n]
        out += [token] + guesses[:n]
        yield [token] + guesses[:n], len(guesses)


def generate(model, ids, draft=None, max_new_tokens=128, eos_id=None):
    """The tokens greedy decoding writes after ids, up to the end of the text, and the passes it took."""
    out, count = [], 0
    for wrote, _ in passes(model, ids, draft):
        count += 1
        if eos_id in wrote:
            return out + wrote[:wrote.index(eos_id)], count
        out += wrote
        if len(out) >= max_new_tokens:
            return out[:max_new_tokens], count
    return out, count


def main():
    import argparse

    from agent import prompt
    from checkpoint import load_checkpoint
    from model import pick_device

    parser = argparse.ArgumentParser(description="Watch a role write up to its first call, a pass at a time, with a "
                                                 "draft's guesses.")
    parser.add_argument("checkpoint")
    parser.add_argument("task")
    parser.add_argument("--role", default="graph")
    parser.add_argument("--draft", default="lookup", help="lookup, a small model's checkpoint, or none")
    parser.add_argument("-k", type=int, default=4, help="tokens to guess in each pass")
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    draft = None
    if args.draft == "lookup":
        draft = Lookup(args.k)
    elif args.draft != "none":
        draft = Draft(load_checkpoint(args.draft, device)[0].eval(), args.k)
    ids = tokenizer.encode(prompt(args.role, args.task))
    print(f"{prompt(args.role, args.task)}")
    out, guessed, kept = [], 0, 0
    # Each pass on a line: the model's own token in brackets, then the guesses it kept, up to the end of the call.
    for count, (wrote, offered) in enumerate(passes(model, ids, draft), start=1):
        guessed, kept = guessed + offered, kept + len(wrote) - 1
        shown = []
        for token in wrote:
            out.append(token)
            shown.append(tokenizer.decode([token]))
            if tokenizer.decode(out).endswith(">>") or token == tokenizer.eot_id:
                break
        print(f"{count:>4}  [{shown[0]}]{''.join(shown[1:])}")
        if tokenizer.decode(out).endswith(">>") or tokenizer.eot_id in out or len(out) > 200:
            break
    print(f"\n{len(out)} tokens in {count} passes, {kept} of {guessed} guesses kept")


if __name__ == "__main__":
    main()
