import re
from collections import Counter

# A run of letters, digits or bytes of multi-byte characters, or of punctuation, each with an optional
# space in front, or a run of whitespace: the chunks merges never cross.
CHUNK = re.compile(rb" ?[A-Za-z0-9\x80-\xff]+| ?[^A-Za-z0-9\x80-\xff\s]+|\s+")


def pretokenize(text):
    return CHUNK.findall(text)


def merge(ids, pair, merged):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(text, n_merges):
    counts = Counter(pretokenize(text))
    words = [(list(chunk), n) for chunk, n in counts.items()]
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for ids, n in words:
            for pair in zip(ids, ids[1:]):
                pairs[pair] += n
        if not pairs:
            break
        # The most frequent pair, and of those, the smallest.
        pair = min(pairs, key=lambda p: (-pairs[p], p))
        merged = 256 + len(merges)
        merges.append(pair)
        words = [(merge(ids, pair, merged), n) for ids, n in words]
    return merges


def encode(text, merges):
    ranks = {pair: r for r, pair in enumerate(merges)}
    out = []
    for chunk in pretokenize(text):
        ids = list(chunk)
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: ranks.get(p, len(ranks)))
            if pair not in ranks:
                break
            ids = merge(ids, pair, 256 + ranks[pair])
        out.extend(ids)
    return out
