import os

import numpy as np
import pytest

import bpe
import bpe_reference as reference
from data import DATA_DIR, load_tokenizer
from tokenizer import EOT

DATA = os.path.join(DATA_DIR, "train.txt")
SAMPLE = open(DATA, "rb").read(500_000) if os.path.exists(DATA) else b"Once upon a time, there was a little girl. " * 2000


def test_pretokenize_matches_the_reference():
    assert [bytes(chunk) for chunk in bpe.pretokenize(SAMPLE)] == reference.pretokenize(SAMPLE)


def test_merges_match_the_reference():
    assert [tuple(m) for m in bpe.train(SAMPLE, 200)] == reference.train(SAMPLE, 200)


def test_ids_match_the_reference():
    merges = reference.train(SAMPLE, 200)
    text = SAMPLE[-50_000:]
    assert bpe.Encoder(merges).encode(text).tolist() == reference.encode(text, merges)


def test_a_small_cache_gives_the_same_ids():
    # Two chunks at most, so that most chunks push another out of the cache before they're seen again.
    merges = reference.train(SAMPLE, 200)
    text = SAMPLE[-50_000:]
    assert bpe.Encoder(merges, cache_size=2).encode(text).tolist() == reference.encode(text, merges)


@pytest.mark.parametrize("text", ["Once upon a time, Lily's dog ran.", "Café, naïve, 日本語, emoji 🙂", "  two  spaces\n\nand lines\t"])
def test_round_trip(text):
    merges = bpe.train(SAMPLE, 200)
    ids = bpe.Encoder(merges).encode(text.encode("utf-8")).tolist()
    pieces = [bytes([b]) for b in range(256)]
    for a, b in merges:
        pieces.append(pieces[a] + pieces[b])
    assert b"".join(pieces[i] for i in ids).decode("utf-8") == text


def test_special_tokens_split_the_text():
    merges = reference.train(SAMPLE, 200)
    encoder = bpe.Encoder(merges, {EOT.encode(): 9999})
    a, b = b"Once upon a time", b" there was a cat."
    ids = encoder.encode(a + EOT.encode() + b + EOT.encode() + EOT.encode()).tolist()
    assert ids == reference.encode(a, merges) + [9999] + reference.encode(b, merges) + [9999, 9999]


WEB = [os.path.join(DATA_DIR, name) for name in ("web-merges.txt", "valid-web.txt", "valid-web.bin")]


@pytest.mark.skipif(not all(map(os.path.exists, WEB)), reason="needs python data.py --web")
def test_the_held_out_pages_encode_as_they_did():
    # valid-web.bin is what data.py --web wrote in episode 19, with the tokenizer as it was then: the same ids, all
    # of them, or the models trained on the web would read their pages differently from now on.
    tokenizer = load_tokenizer(bpe=True, merges="web-merges.txt")
    with open(WEB[1], "rb") as f:
        ids = tokenizer.encode_bytes(f.read())
    assert np.array_equal(ids, np.fromfile(WEB[2], dtype=np.uint16))
