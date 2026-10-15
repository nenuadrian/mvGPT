import os

import pytest
import torch
from torch.nn import functional as F

triton = pytest.importorskip("triton", reason="Triton runs on Linux, with NVIDIA (or AMD) GPUs")
from flash import flash_attention  # noqa: E402

# On an NVIDIA GPU; or, slowly, on the CPU, with Triton's interpreter (TRITON_INTERPRET=1).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else None
DTYPES = [torch.bfloat16, torch.float16] if DEVICE == "cuda" else [torch.float32, torch.float16]
LENGTHS = [1, 77, 128, 300, 1024] if DEVICE == "cuda" else [1, 77, 128, 200]
pytestmark = pytest.mark.skipif(DEVICE is None, reason="no NVIDIA GPU, and not TRITON_INTERPRET=1")


def qkv(T, dtype):
    torch.manual_seed(0)
    return [torch.randn(2, 3, T, 64, device=DEVICE, dtype=dtype) for _ in range(3)]


def pytorch(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", LENGTHS)
def test_matches_pytorch(T, dtype):
    # Against PyTorch's attention in float64, ours may be off by rounding, but by no more than twice as
    # much as PyTorch's own attention is in the same precision: the test FlashAttention's authors use.
    q, k, v = qkv(T, dtype)
    exact = pytorch(q.double(), k.double(), v.double())
    their_error = (pytorch(q, k, v).double() - exact).abs().max().item()
    our_error = (flash_attention(q, k, v).double() - exact).abs().max().item()
    assert our_error <= 2 * their_error + 1e-5, f"{our_error:.2e} against PyTorch's {their_error:.2e}"
