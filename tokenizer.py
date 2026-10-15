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


BPE_SPECIALS_SPLIT = re.compile("(" + re.escape(PAD) + "|" + re.escape(EOT) + ")")


class BPETokenizer:
    def __init__(self, merges):
        import bpe  # the C++ module: python setup.py build_ext --inplace

        self.merges = [tuple(m) for m in merges]
        self.encoder = bpe.Encoder(self.merges)
        # Each token's bytes: the 256 single bytes, then, for each merge, its pair's bytes joined.
        self.token_bytes = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        self.pad_id = len(self.token_bytes)
        self.eot_id = self.pad_id + 1
        # Any text is bytes, and each byte is a token, so there's no unknown token.
        self.unk_id = None
        self.id_to_token = [t.decode("utf-8", errors="replace") for t in self.token_bytes] + [PAD, EOT]

    def __len__(self):
        return len(self.id_to_token)

    def encode(self, text):
        ids = []
        for piece in BPE_SPECIALS_SPLIT.split(text):
            if piece == PAD:
                ids.append(self.pad_id)
            elif piece == EOT:
                ids.append(self.eot_id)
            elif piece:
                ids.extend(self.encoder.encode(piece.encode("utf-8")))
        return ids

    def decode(self, ids, skip_special=True):
        pieces = []
        for i in ids:
            if i < len(self.token_bytes):
                pieces.append(self.token_bytes[i])
            elif not skip_special:
                pieces.append(self.id_to_token[i].encode("utf-8"))
        # A token can end in the middle of a character, so the joined bytes are decoded at the end.
        return b"".join(pieces).decode("utf-8", errors="replace")
