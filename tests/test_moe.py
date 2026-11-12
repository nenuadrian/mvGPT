import torch

from model import GPT, MLP, MODERN, GPTConfig, MoE


def tiny(n_expert=4, top_k=2):
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_embd=32, dropout=0.0,
                       n_expert=n_expert, expert_top_k=top_k, **MODERN)
    return GPT(config).eval()


def test_without_experts_each_block_keeps_its_mlp():
    model = tiny(n_expert=0)
    assert all(type(block.mlp) is MLP for block in model.blocks)
    assert not any("router" in name for name, _ in model.named_parameters())


def test_each_token_gets_the_weighted_sum_of_its_top_k_experts():
    moe = tiny().blocks[0].mlp
    x = torch.randn(2, 5, 32)
    out = moe(x)
    for b in range(2):
        for t in range(5):
            # One token at a time, written out: its scores, its two best experts, their weights adding up to 1.
            probs = torch.softmax(moe.router(x[b, t]), dim=-1)
            best = probs.topk(2).indices
            weights = probs[best] / probs[best].sum()
            expected = sum(w * moe.expert(e, x[b, t]) for w, e in zip(weights, best))
            assert (out[b, t] - expected).abs().max() < 1e-5


def test_an_expert_costs_what_the_mlp_cost_divided_by_top_k():
    moe, mlp = tiny().blocks[0].mlp, tiny(n_expert=0).blocks[0].mlp
    per_expert = moe.gate[0].numel() + moe.up[0].numel() + moe.down[0].numel()
    assert per_expert * moe.top_k == sum(p.numel() for p in mlp.parameters())


def test_the_balancing_loss_is_one_when_even_and_grows_when_one_expert_gets_all_the_tokens():
    moe = tiny().blocks[0].mlp
    x = torch.randn(4, 8, 32)
    with torch.no_grad():
        moe.router.weight.zero_()
    moe(x)
    assert abs(moe.aux_loss.item() - 1.0) < 1e-5
    with torch.no_grad():
        # Channel 0 is 1 on each token, and the router scores expert 0 at 100 from it, and the others at 0.
        x[..., 0] = 1.0
        moe.router.weight[0, 0] = 100.0
    moe(x)
    assert moe.load.tolist()[0] == 4 * 8
    # Expert 0 is one of the two for each token, so it gets half the pairs: 4 x 0.5 x 1, over the 2 picked.
    assert abs(moe.aux_loss.item() - moe.n_expert / moe.top_k) < 1e-3
