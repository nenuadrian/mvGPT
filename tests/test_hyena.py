import pytest
import torch

from hyena import Hyena, direct_conv, fft_conv
from model import GPT, GPTConfig, KVCache, kv_cache_bytes


def tiny(**kwargs):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=3, n_head=4, n_embd=32, dropout=0.0, rope=True,
                       mixer="hyena", **kwargs)
    return GPT(config).eval()


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


@pytest.mark.parametrize("attention_layers", [(), (1,)])
def test_writing_through_the_cache_matches_running_the_whole_sequence(attention_layers):
    model = tiny(attention_layers=attention_layers)
    idx = torch.randint(50, (2, 12))
    full, _ = model(idx)
    cache = KVCache(model.config.n_layer)
    steps = [model(idx[:, :8], cache=cache)[0]] + [model(idx[:, t : t + 1], cache=cache)[0] for t in range(8, 12)]
    assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-4
    assert len(cache) == 12


def test_a_hybrid_keeps_attention_in_the_blocks_it_names():
    model = tiny(attention_layers=(1,))
    assert [hasattr(block, "attn") for block in model.blocks] == [False, True, False]
    assert [hasattr(block, "hyena") for block in model.blocks] == [True, False, True]


def test_the_cache_keeps_one_number_a_channel_for_a_hyena_block():
    model = tiny(attention_layers=(1,))
    cache = KVCache(model.config.n_layer)
    model(torch.randint(50, (1, 10)), cache=cache)
    grows = [t for t in cache.keys + cache.values if t is not None] + [s["z"] for s in cache.hyena if s]
    assert sum(t.numel() * t.element_size() for t in grows) == kv_cache_bytes(model.config, 10)
    # Two Hyena blocks of 32 channels and one attention block's keys and values: 2 x 32 + 2 x 32 numbers a token.
    assert kv_cache_bytes(model.config, 1) == (2 * 32 + 2 * 32) * 4


def test_generate_writes_past_the_window_with_the_cache():
    model = tiny()
    out = model.generate(torch.randint(50, (1, 4)), max_new_tokens=40, top_k=5)
    assert out.shape == (1, 44)
