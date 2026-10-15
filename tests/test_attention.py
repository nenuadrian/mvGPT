import pytest
import torch
from torch.nn import functional as F

from attention import attention_by_hand, flash_attention_pytorch

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


def qkv(T, device):
    torch.manual_seed(0)
    return [torch.randn(2, 3, T, 64, device=device) for _ in range(3)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T", [1, 100, 256, 1000])
def test_by_hand_matches_pytorch(device, T):
    q, k, v = qkv(T, device)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (attention_by_hand(q, k, v) - expected).abs().max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("T", [1, 100, 256, 1000])
def test_flash_in_pytorch_matches_pytorch(device, T):
    q, k, v = qkv(T, device)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (flash_attention_pytorch(q, k, v) - expected).abs().max() < 1e-5
