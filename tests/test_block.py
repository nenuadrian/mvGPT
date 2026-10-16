import torch
from torch.nn import functional as F

from model import MLP, CausalSelfAttention, GPTConfig, RMSNorm


def test_rmsnorm_matches_pytorch_and_brings_each_token_to_a_root_mean_square_of_1():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 384) * 3 + 1
    norm = RMSNorm(384)
    y = norm(x)
    assert (y - F.rms_norm(x, (384,), norm.weight, eps=norm.eps)).abs().max() < 1e-5
    # With its gains at 1, each token's channels come out with a root mean square of 1.
    assert (y.pow(2).mean(dim=-1).sqrt() - 1).abs().max() < 1e-4


def test_swiglu_has_as_many_weights_as_gelus_mlp():
    counts = []
    for swiglu in (False, True):
        mlp = MLP(GPTConfig(swiglu=swiglu, bias=False))
        counts.append(sum(p.numel() for p in mlp.parameters()))
    # GELU's two matrices of 384 x 1,536, and SwiGLU's three of 384 x 1,024: 1,179,648 weights each.
    assert counts == [1_179_648, 1_179_648]


def test_qk_norm_keeps_attention_from_growing_with_the_weights():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 384)
    for qk_norm in (False, True):
        attn = CausalSelfAttention(GPTConfig(qk_norm=qk_norm, bias=False, dropout=0.0), layer=0)
        before = attn(x)
        with torch.no_grad():
            attn.q_proj.weight *= 10
            attn.k_proj.weight *= 10
        # Ten times the weights make the scores a hundred times larger, unless the queries and keys are normalised.
        assert ((attn(x) - before).abs().max() < 1e-4) == qk_norm
