import pytest
import torch

from model import GPT, MODERN, GPTConfig, KVCache


def tiny(n_kv_head, modern):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=6, n_embd=48, dropout=0.0, rope=True,
                       n_kv_head=n_kv_head, **(MODERN if modern else {}))
    return GPT(config).eval()


@pytest.mark.parametrize("modern", [False, True])
def test_gqa_is_multi_head_attention_with_each_key_and_value_head_repeated_for_its_group(modern):
    gqa, mha = tiny(2, modern), tiny(None, modern)
    state = gqa.state_dict()
    for name, w in state.items():
        if ".k_proj." in name or ".v_proj." in name:
            # 2 heads of 8 channels, each repeated for the 3 query heads of its group: 6 heads, as in mha.
            state[name] = w.view(2, 8, *w.shape[1:]).repeat_interleave(3, dim=0).reshape(48, *w.shape[1:])
    mha.load_state_dict(state)
    idx = torch.randint(50, (2, 16))
    assert (gqa(idx)[0] - mha(idx)[0]).abs().max() < 1e-5


@pytest.mark.parametrize("n_kv_head", [2, 1])
def test_the_cache_holds_fewer_heads_and_matches_running_the_whole_sequence(n_kv_head):
    model = tiny(n_kv_head, modern=True)
    idx = torch.randint(50, (2, 12))
    full, _ = model(idx)
    cache = KVCache(model.config.n_layer)
    steps = [model(idx[:, :8], cache=cache)[0]] + [model(idx[:, t : t + 1], cache=cache)[0] for t in range(8, 12)]
    assert (torch.cat(steps, dim=1) - full).abs().max() < 1e-5
    # Batch, key and value heads, tokens, channels per head.
    assert cache.keys[0].shape == (2, n_kv_head, 12, 8)
