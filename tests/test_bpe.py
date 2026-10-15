import os

import pytest

import bpe
import bpe_reference as reference

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train.txt")
SAMPLE = open(DATA, "rb").read(500_000) if os.path.exists(DATA) else b"Once upon a time, there was a little girl. " * 2000


def test_pretokenize_matches_the_reference():
    assert [bytes(chunk) for chunk in bpe.pretokenize(SAMPLE)] == reference.pretokenize(SAMPLE)


def test_merges_match_the_reference():
    assert [tuple(m) for m in bpe.train(SAMPLE, 200)] == reference.train(SAMPLE, 200)


def test_ids_match_the_reference():
    merges = reference.train(SAMPLE, 200)
    text = SAMPLE[-50_000:]
    assert bpe.Encoder(merges).encode(text) == reference.encode(text, merges)


@pytest.mark.parametrize("text", ["Once upon a time, Lily's dog ran.", "Café, naïve, 日本語, emoji 🙂", "  two  spaces\n\nand lines\t"])
def test_round_trip(text):
    merges = bpe.train(SAMPLE, 200)
    ids = bpe.Encoder(merges).encode(text.encode("utf-8"))
    pieces = [bytes([b]) for b in range(256)]
    for a, b in merges:
        pieces.append(pieces[a] + pieces[b])
    assert b"".join(pieces[i] for i in ids).decode("utf-8") == text
