import torch

from constrain import Names, pick, vocabulary
from tokenizer import BPETokenizer


def tokenizer_with(*pieces):
    # The 256 bytes, and a token for each piece, merged from the piece without its last byte and that byte.
    merges, ids = [], {bytes([b]): b for b in range(256)}
    for piece in pieces:
        piece = piece.encode()
        for end in range(2, len(piece) + 1):
            if piece[:end] not in ids:
                merges.append((ids[piece[:end - 1]], piece[end - 1]))
                ids[piece[:end]] = 255 + len(merges)
    return BPETokenizer(merges)


def texts(tokenizer, ids):
    # The allowed tokens' text, in order, to compare with a list.
    return sorted(tokenizer.id_to_token[i] for i in ids)


def test_the_tokens_a_name_allows():
    tokenizer = tokenizer_with("Ok", "ak", "ap", "api", "i'})")
    names = Names(vocabulary(tokenizer), ["Okapi", "Oak Titmouse"])
    assert texts(tokenizer, names.allowed(b"MATCH (s:Species {name: '")) == ["O", "Ok"]
    # After the O, the k of Okapi, and the a and the ak of Oak Titmouse.
    assert texts(tokenizer, names.allowed(b"{name: 'O")) == ["a", "ak", "k"]
    assert texts(tokenizer, names.allowed(b"{name: 'Ok")) == ["a", "ap", "api"]
    # At the end of a name, the quote closes it, and a token can carry on past it.
    assert texts(tokenizer, names.allowed(b"{name: 'Okap")) == ["i", "i'", "i'}", "i'})"]
    assert texts(tokenizer, names.allowed(b"{name: 'Okapi")) == ["'"]


def test_outside_a_name_anything_goes():
    tokenizer = tokenizer_with("ak")
    names = Names(vocabulary(tokenizer), ["Okapi"])
    assert names.allowed(b"MATCH (s:Species") is None
    assert names.allowed(b"{name: 'Okapi'}) RETURN s.weight_g") is None
    # A name that's off the list already is left as it is.
    assert names.allowed(b"{name: 'Oak") is None


def test_double_quotes_for_an_apostrophe():
    tokenizer = tokenizer_with()
    names = Names(vocabulary(tokenizer), ["Finsch's Francolin"])
    assert texts(tokenizer, names.allowed(b'{name: "Finsch')) == ["'"]
    assert texts(tokenizer, names.allowed(b"{name: \"Finsch's Francolin")) == ['"']
    # In single quotes, the apostrophe would close the name early, as 'Finsch', so it isn't allowed.
    names = Names(vocabulary(tokenizer), ["Finsch's Francolin", "Finschia"])
    assert texts(tokenizer, names.allowed(b"{name: 'Finsch")) == ["i"]


def test_a_name_in_bytes():
    # é is two bytes, and a token can hold the first alone.
    tokenizer = tokenizer_with()
    names = Names(vocabulary(tokenizer), ["Réunion Kestrel"])
    assert names.allowed("{name: 'R".encode()) == [0xC3]
    assert names.allowed("{name: 'R".encode() + b"\xc3") == [0xA9]


def test_the_mask():
    logits = torch.tensor([0.0, 5.0, 2.0, 1.0])
    assert pick(logits, None) == 1
    assert pick(logits, [2, 3]) == 2
