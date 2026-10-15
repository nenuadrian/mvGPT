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
    shape = (2, 3, T, 64)
    return [torch.randn(shape, device=DEVICE, dtype=dtype, requires_grad=True) for _ in range(4)]


def attend(fn, q, k, v, grad):
    # The output, and the gradients of q, k and v for this gradient of it.
    out = fn(q, k, v)
    return [out] + list(torch.autograd.grad(out, (q, k, v), grad))


def pytorch(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", LENGTHS)
def test_matches_pytorch(T, dtype):
    # Against PyTorch's attention in float64, ours may be off by rounding, but by no more than twice as
    # much as PyTorch's own attention is in the same precision: the test FlashAttention's authors use.
    q, k, v, grad = qkv(T, dtype)
    exact = attend(pytorch, *(x.detach().double().requires_grad_() for x in (q, k, v)), grad.double())
    theirs = attend(pytorch, q, k, v, grad)
    ours = attend(flash_attention, q, k, v, grad)
    for name, e, t, o in zip(["output", "dq", "dk", "dv"], exact, theirs, ours):
        their_error = (t.double() - e).abs().max().item()
        our_error = (o.double() - e).abs().max().item()
        assert our_error <= 2 * their_error + 1e-5, f"{name}: {our_error:.2e} against PyTorch's {their_error:.2e}"
