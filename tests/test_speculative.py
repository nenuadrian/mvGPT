import torch

from model import GPT, GPTConfig, KVCache
from speculative import Draft, Lookup, generate, passes


def tiny(seed):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=50, block_size=128, n_layer=2, n_head=2, n_embd=32, rope=True)).eval()


@torch.no_grad()
def greedy(model, ids, n):
    # A token at a time, with no cache: the text speculative decoding has to match.
    out = []
    for _ in range(n):
        logits, _ = model(torch.tensor([ids + out]))
        out.append(logits[0, -1].argmax().item())
    return out


def test_the_text_is_greedy_decodings():
    model, ids = tiny(0), [1, 2, 3, 4, 1, 2, 3]
    want = greedy(model, ids, 40)
    # No draft, guesses from the text, a draft that's another model, and the model as its own draft.
    for draft in (None, Lookup(), Draft(tiny(1)), Draft(model)):
        assert generate(model, ids, draft, max_new_tokens=40)[0] == want


def test_a_draft_that_is_the_model_keeps_each_guess():
    model = tiny(0)
    for wrote, guessed in passes(model, [1, 2, 3], Draft(model, k=4)):
        assert (guessed, len(wrote)) == (4, 5)
        break
    out, count = generate(model, [1, 2, 3], Draft(model, k=4), max_new_tokens=40)
    assert count == 40 // 5  # five tokens a pass, after the prompt's


def test_lookup_guesses_what_came_after():
    assert Lookup(k=3).guess([5, 6, 7, 8, 9, 5, 6]) == [7, 8, 9]
    assert Lookup(k=3).guess([5, 6, 7, 6]) == [7, 6]  # the last token alone, when the longer tails weren't written before
    assert Lookup(k=3).guess([5, 6, 7]) == []


def test_trimming_the_cache():
    model, cache = tiny(0), KVCache(2)
    model(torch.tensor([[1, 2, 3, 4, 5]]), cache=cache)
    cache.trim(2)
    assert len(cache) == 3 and cache.position == 3
    # The logits after the trimmed cache are the ones the first three tokens give.
    assert torch.allclose(model(torch.tensor([[9]]), cache=cache)[0], model(torch.tensor([[1, 2, 3, 9]]))[0][:, -1:],
                          atol=1e-5)
