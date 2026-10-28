import torch

from interpret import HeadsOff, attention, lens
from model import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_head=2, n_embd=32, rope=True)).eval()


def test_the_weights():
    model, ids = tiny(), [5, 6, 7, 8, 9]
    before = model(torch.tensor([ids]))[0]
    w = attention(model, ids)
    assert w.shape == (2, 2, 5, 5)
    # Each token shares its attention out over itself and the tokens before it, and none after.
    assert torch.allclose(w.sum(-1), torch.ones(2, 2, 5))
    assert w[0, 0, 1, 2:].sum() == 0
    # Writing attention out by hand gives the logits PyTorch's own does.
    for block in model.blocks:
        block.attn.keep = True
    assert torch.allclose(model(torch.tensor([ids]))[0], before, atol=1e-5)


def test_the_lens_ends_on_the_model():
    model, ids = tiny(), [5, 6, 7]
    p = lens(model, ids)
    assert p.shape == (3, 3, 300)  # the embeddings, then two blocks
    assert torch.allclose(p[-1], torch.softmax(model(torch.tensor([ids]))[0][0], dim=-1), atol=1e-5)


def test_heads_off():
    model, x = tiny(), torch.tensor([[5, 6, 7, 8]])
    before = model(x)[0]
    with HeadsOff(model, [(1, 0)]):
        assert not torch.allclose(model(x)[0], before)
    assert torch.allclose(model(x)[0], before)  # back on, once it's closed
