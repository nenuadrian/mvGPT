import re
from collections import Counter

EOT = "<|endoftext|>"
PAD = "[PAD]"
UNK = "[UNK]"
SPECIALS = [UNK, PAD, EOT]

# A word, or a run of punctuation; whitespace matches neither, so it's dropped.
PRETOKENIZE = re.compile(r"\w+|[^\w\s]+")
# The group makes re.split keep the special tokens it splits on.
SPECIALS_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


def pretokenize(text):
    return PRETOKENIZE.findall(text)


def build_vocab(paths, size, min_count=2):
    counts = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                counts.update(pretokenize(line.replace(EOT, " ")))
    # Most frequent first, and ties in alphabetical order, so the same data gives the same ids.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    words = [word for word, count in ranked if count >= min_count]
    return SPECIALS + words[: size - len(SPECIALS)], counts


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;:]\S*)", r"\1", text)
    text = re.sub(r"(\w) ' (\w)", r"\1'\2", text)
    # An opening quote goes against the word after it.
    text = re.sub(r'(^|\s)" ', r'\1"', text)
    return text


class WordTokenizer:
    def __init__(self, vocab):
        self.id_to_token = list(vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}
        self.unk_id = self.token_to_id[UNK]
        self.pad_id = self.token_to_id[PAD]
        self.eot_id = self.token_to_id[EOT]

    def __len__(self):
        return len(self.id_to_token)

    def encode(self, text):
        ids = []
        for piece in SPECIALS_SPLIT.split(text):
            if piece in SPECIALS:
                ids.append(self.token_to_id[piece])
            else:
                ids.extend(self.token_to_id.get(t, self.unk_id) for t in pretokenize(piece))
        return ids

    def decode(self, ids, skip_special=True):
        tokens = [self.id_to_token[i] for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in SPECIALS]
        return detokenize(tokens)
