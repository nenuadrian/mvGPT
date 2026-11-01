import pytest
import torch

from hyena import Hyena, direct_conv, fft_conv
from model import GPTConfig


@pytest.mark.parametrize("T", [1, 7, 32, 100])
def test_the_fft_gives_the_sums_written_out(T):
    torch.manual_seed(0)
    u, h = torch.randn(2, 5, T), torch.randn(5, T)
    assert (fft_conv(u, h) - direct_conv(u, h)).abs().max() < 1e-4


def test_an_output_never_depends_on_tokens_after_it():
    torch.manual_seed(0)
    layer = Hyena(GPTConfig(n_embd=32, block_size=64, dropout=0.0), layer=0)
    x = torch.randn(1, 64, 32, requires_grad=True)
    y = layer(x)
    # The gradient of output 40 with respect to every input: not zero up to token 40, and after it zero, but for
    # the FFT's rounding, under a millionth of the rest.
    (grad,) = torch.autograd.grad(y[0, 40].sum(), x)
    assert grad[0, :41].abs().sum(dim=-1).min() > 0
    assert grad[0, 41:].abs().max() < 1e-6 * grad[0, :41].abs().max()


def test_each_channel_is_a_matrix_of_gates_and_the_filter_applied_to_its_values():
    torch.manual_seed(0)
    layer = Hyena(GPTConfig(n_embd=32, block_size=64, dropout=0.0), layer=0)
    x = torch.randn(1, 20, 32)
    _, _, v = layer.streams(x)
    # y = D(x2) (S(h) + b I) D(x1) v for each channel, before the output projection.
    y = torch.einsum("cts,cs->ct", layer.matrices(x), v[0])
    assert (layer.o_proj(y.T) - layer(x)[0]).abs().max() < 1e-4
    # Nothing above the diagonal: no token looks ahead.
    assert layer.matrices(x).triu(diagonal=1).abs().max() == 0


def test_the_filter_has_the_same_weights_for_any_window():
    short = Hyena(GPTConfig(n_embd=32, block_size=256), layer=0).filter
    long = Hyena(GPTConfig(n_embd=32, block_size=4096), layer=0).filter
    assert sum(p.numel() for p in short.parameters()) == sum(p.numel() for p in long.parameters())
