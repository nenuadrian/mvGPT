import math

import torch

from diffusion import add_noise, diffusion_loss
from model import GPT, GPTConfig

MASK = 49  # the last of 50 ids


def tiny(causal):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0, rope=True,
                       causal=causal)
    return GPT(config).eval()


def test_a_causal_model_cant_see_ahead_and_a_diffusion_model_can():
    x = torch.randint(MASK, (1, 32))
    later = x.clone()
    later[0, 20] = (later[0, 20] + 1) % MASK  # change a token near the end
    for causal in (True, False):
        model = tiny(causal)
        before = (model(x)[0][0, :20] - model(later)[0][0, :20]).abs().max().item()
        # The outputs before token 20 change only when the model can see token 20.
        assert (before == 0) == causal


def test_noise_hides_about_the_rate_and_never_the_first_token():
    torch.manual_seed(0)
    x = torch.randint(MASK, (4, 1000))
    noisy, hidden = add_noise(x, torch.tensor([0.0, 0.1, 0.5, 0.9]), MASK)
    assert not hidden[:, 0].any()
    assert ((noisy == MASK) == hidden).all()
    assert (noisy[~hidden] == x[~hidden]).all()
    rates = hidden[:, 1:].float().mean(1)
    assert (rates - torch.tensor([0.0, 0.1, 0.5, 0.9])).abs().max() < 0.05


def test_an_untrained_model_scores_about_ln_vocab():
    # Before training, every token is about as likely as any other, at any amount of noise.
    torch.manual_seed(0)
    loss = diffusion_loss(tiny(False), torch.randint(MASK, (256, 32)), MASK).item()
    assert abs(loss - math.log(50)) < 0.3