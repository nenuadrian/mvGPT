import torch

from hub import from_gpt2, to_gpt2
from model import GPT, GPTConfig
from tokenizer import SPECIALS, WordTokenizer


def test_gpt2_and_back():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=8)
    ours, tokenizer = GPT(config).eval(), WordTokenizer(SPECIALS + ["a", "b"])
    gpt2 = to_gpt2(ours, tokenizer)
    # A fresh model, so that nothing can pass by being left as it was.
    back = from_gpt2(gpt2, GPT(config).eval())
    for name, weight in ours.state_dict().items():
        assert torch.equal(weight, back.state_dict()[name]), name
    x = torch.randint(0, config.vocab_size, (2, 10))
    assert torch.allclose(ours(x)[0], gpt2(x).logits, atol=1e-5)
