import math

import pytest
import torch

from model import SCALINGS, GPT, GPTConfig, KVCache, rope_speeds, rotary, stretch


def tiny(**options):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0, rope=True, **options)
    return GPT(config).eval()


def test_position_interpolation_turns_position_4p_as_far_as_p():
    cos, sin = rotary(0, 4096, 64, "cpu")
    pi_cos, pi_sin = rotary(0, 4096, 64, "cpu", "pi", 4, 1024)
    assert torch.allclose(pi_cos[::4][:1024], cos[:1024], atol=1e-4)
    assert torch.allclose(pi_sin[::4][:1024], sin[:1024], atol=1e-4)


def test_ntk_keeps_the_fastest_pair_and_slows_the_slowest_by_the_factor():
    speeds, ntk = rope_speeds(64, "cpu"), rope_speeds(64, "cpu", "ntk", 4, 1024)
    assert ntk[0] == speeds[0]
    assert torch.isclose(ntk[-1], speeds[-1] / 4)
    # In between, each pair is slowed by some, and never by more than the factor.
    assert ((ntk <= speeds) & (ntk >= speeds / 4 - 1e-9)).all()


@pytest.mark.parametrize("scaling", ["parts", "yarn"])
def test_by_parts_keeps_the_fast_pairs_and_slows_the_slow_ones_by_the_factor(scaling):
    speeds, parts = rope_speeds(64, "cpu"), rope_speeds(64, "cpu", scaling, 4, 1024)
    turns = 1024 * speeds / (2 * math.pi)
    assert torch.equal(parts[turns >= 32], speeds[turns >= 32])
    assert torch.allclose(parts[turns <= 1], speeds[turns <= 1] / 4)
    between = (turns > 1) & (turns < 32)
    assert between.any() and ((parts[between] < speeds[between]) & (parts[between] > speeds[between] / 4)).all()


def test_yarn_lengthens_queries_and_keys_by_its_temperature():
    cos, sin = rotary(0, 100, 64, "cpu", "yarn", 4, 1024)
    m = 0.1 * math.log(4) + 1
    assert torch.allclose(cos**2 + sin**2, torch.full_like(cos, m**2))


@pytest.mark.parametrize("scaling", SCALINGS)
def test_stretching_by_1_changes_nothing(scaling):
    model, idx = tiny(), torch.randint(50, (2, 20))
    stretched = GPT(stretch(model.config, scaling, 1)).eval()
    stretched.load_state_dict(model.state_dict())
    assert stretched.config.block_size == 32
    assert (stretched(idx)[0] - model(idx)[0]).abs().max() < 1e-5


@pytest.mark.parametrize("scaling", SCALINGS)
def test_a_stretched_model_reads_past_its_window_and_the_cache_matches_the_whole_sequence(scaling):
    model = tiny()
    stretched = GPT(stretch(model.config, scaling, 4)).eval()
    stretched.load_state_dict(model.state_dict())
    assert stretched.config.block_size == 128
    idx = torch.randint(50, (2, 100))
    full, _ = stretched(idx)
    cache = KVCache(stretched.config.n_layer)
    steps = [stretched(idx[:, :90], cache=cache)[0]] + [stretched(idx[:, t : t + 1], cache=cache)[0] for t in range(90, 100)]
    assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-4


def test_scaling_needs_rotary_embeddings_and_a_known_method():
    with pytest.raises(ValueError):
        GPT(GPTConfig(rope=False, rope_scaling="yarn"))
    with pytest.raises(ValueError):
        GPT(GPTConfig(rope=True, rope_scaling="linear"))
