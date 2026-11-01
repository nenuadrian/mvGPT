import pytest
import torch

from hyena import direct_conv, fft_conv


@pytest.mark.parametrize("T", [1, 7, 32, 100])
def test_the_fft_gives_the_sums_written_out(T):
    torch.manual_seed(0)
    u, h = torch.randn(2, 5, T), torch.randn(5, T)
    assert (fft_conv(u, h) - direct_conv(u, h)).abs().max() < 1e-4
