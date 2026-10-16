import torch
from torch.nn import functional as F

from model import GPT, MLP, MODERN, CausalSelfAttention, GPTConfig, KVCache, RMSNorm


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


def test_the_kv_cache_matches_running_the_whole_sequence_with_the_modern_block():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0, rope=True, **MODERN)
    model = GPT(config).eval()
    idx = torch.randint(50, (2, 12))
    full, _ = model(idx)
    cache = KVCache(config.n_layer)
    # The first 8 tokens at once, then one at a time, as generate does.
    steps = [model(idx[:, :8], cache=cache)[0]] + [model(idx[:, t : t + 1], cache=cache)[0] for t in range(8, 12)]
    assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-5


def test_a_config_saved_before_these_options_builds_the_original_block():
    # A checkpoint's config from before this episode, without the new fields.
    old = {"vocab_size": 10_000, "block_size": 256, "n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.1,
           "bias": True, "rope": False, "causal": True}
    model = GPT(GPTConfig(**old))
    assert isinstance(model.norm, torch.nn.LayerNorm) and model.blocks[0].mlp.gate_proj is None
    assert model.num_params() == 14_585_856


def test_the_modern_block_drops_the_biases_and_adds_qk_norms_gains():
    original = GPT(GPTConfig(rope=True)).num_params()
    modern = GPT(GPTConfig(rope=True, **MODERN)).num_params()
    # 4,224 biases per block go, and 384 in the final norm, while QK-norm adds 128 gains per block.
    assert original - modern == 6 * 4_224 + 384 - 6 * 128 == 24_960
