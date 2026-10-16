import torch
from torch.nn import functional as F

from model import RMSNorm


def test_rmsnorm_matches_pytorch_and_brings_each_token_to_a_root_mean_square_of_1():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 384) * 3 + 1
    norm = RMSNorm(384)
    y = norm(x)
    assert (y - F.rms_norm(x, (384,), norm.weight, eps=norm.eps)).abs().max() < 1e-5
    # With its gains at 1, each token's channels come out with a root mean square of 1.
    assert (y.pow(2).mean(dim=-1).sqrt() - 1).abs().max() < 1e-4
