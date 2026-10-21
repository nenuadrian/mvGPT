import pytest
import torch

from bakeoff import NAMES, make_optimizers
from model import GPT, GPTConfig
from optimizers import AdamW, Lion, Muon, Sophia, curvature, orthogonalize


def tiny():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0, rope=True)
    return GPT(config)


def train(model, optimizers, steps=5):
    # A few steps on one batch of random tokens, again and again, and the losses on the way. A model
    # can have more than one optimizer, each for some of its weights, so they come as a list.
    torch.manual_seed(1)
    x, y = torch.randint(50, (4, 32)), torch.randint(50, (4, 32))
    losses = []
    for _ in range(steps):
        loss = model(x, y)[1]
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        losses.append(loss.item())
    return losses


def test_our_adamw_matches_pytorchs():
    ours, theirs = tiny(), tiny()
    train(ours, [AdamW(ours.parameters(), lr=1e-2, weight_decay=0.1)])
    train(theirs, [torch.optim.AdamW(theirs.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)])
    for a, b in zip(ours.parameters(), theirs.parameters()):
        assert torch.allclose(a, b, atol=1e-5)


def test_adams_first_step_moves_each_weight_by_the_learning_rate():
    # After one step, m is the gradient and v its square, so the update is the gradient's sign, unless the
    # gradient is so small that eps counts.
    model = tiny()
    before = [p.detach().clone() for p in model.parameters()]
    train(model, [AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)], steps=1)
    for p, b in zip(model.parameters(), before):
        moved = (p - b).abs()
        assert (moved <= 1e-3 * 1.001).all()
        assert torch.allclose(moved[p.grad.abs() > 1e-5], torch.tensor(1e-3), rtol=1e-2)


def test_lion_moves_each_weight_by_the_learning_rate_every_step():
    model = tiny()
    optimizer = Lion(model.parameters(), lr=1e-3, weight_decay=0.0)
    for _ in range(3):
        before = [p.detach().clone() for p in model.parameters()]
        train(model, [optimizer], steps=1)
        for p, b in zip(model.parameters(), before):
            moved = (p - b).abs()
            # Up or down by the learning rate, or not at all where the gradient and the momentum are zero.
            assert torch.allclose(moved[moved > 0], torch.tensor(1e-3), rtol=1e-3)


def test_sophia_moves_no_weight_further_than_the_learning_rate():
    model = tiny()
    optimizer = Sophia(model.parameters(), lr=1e-3, weight_decay=0.0, tokens=128)
    for _ in range(5):
        before = [p.detach().clone() for p in model.parameters()]
        train(model, [optimizer], steps=1)
        curvature(model, torch.randint(50, (4, 32)), optimizer, torch.Generator().manual_seed(0))
        for p, b in zip(model.parameters(), before):
            assert ((p - b).abs() <= 1e-3 * 1.001).all()


def test_sophia_takes_a_newton_step_where_the_curvature_is_large():
    # One weight, with a curvature so large that its step isn't clipped: the momentum over the curvature.
    w = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = Sophia([w], lr=1.0, betas=(0.5, 0.99), rho=0.05, weight_decay=0.0, tokens=100)
    optimizer.state_of(w)["h"].fill_(4.0)
    w.grad = torch.tensor([2.0])
    optimizer.step()
    m = 0.5 * 2.0
    assert torch.isclose(w, torch.tensor([1.0 - m / (0.05 * 100 * 4.0)]))


def test_the_curvature_estimate_is_positive():
    model = tiny()
    optimizer = Sophia(model.parameters())
    curvature(model, torch.randint(50, (4, 32)), optimizer, torch.Generator().manual_seed(0))
    hs = [optimizer.state_of(p)["h"] for p in model.parameters()]
    assert all((h >= 0).all() for h in hs) and sum(h.sum() for h in hs) > 0


def test_orthogonalize_keeps_the_singular_vectors_and_brings_the_values_near_1():
    torch.manual_seed(0)
    # A matrix whose singular values span from 0.1 to 10.
    U, V = torch.linalg.qr(torch.randn(96, 32))[0], torch.linalg.qr(torch.randn(32, 32))[0]
    G = U @ torch.diag(torch.logspace(-1, 1, 32)) @ V.T
    D = U.T @ orthogonalize(G) @ V
    # In G's singular vectors, the result is diagonal: G's directions, with new lengths.
    assert (D - torch.diag(D.diagonal())).abs().max() < 1e-4
    assert ((D.diagonal() > 0.6) & (D.diagonal() < 1.2)).all()


def matrices(model):
    return [p for block in model.blocks for p in block.parameters() if p.dim() == 2]


def test_our_muon_steps_where_pytorchs_does():
    ours, theirs = tiny(), tiny()
    before = [p.detach().clone() for p in matrices(ours)]
    train(ours, [Muon(matrices(ours), lr=1e-2, weight_decay=0.1)])
    train(theirs, [torch.optim.Muon(matrices(theirs), lr=1e-2, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")])
    for a, b, start in zip(matrices(ours), matrices(theirs), before):
        # PyTorch's runs Newton-Schulz in bfloat16, and ours in float32, so they agree to bfloat16's rounding.
        ours_moved, theirs_moved = (a - start).flatten(), (b - start).flatten()
        assert torch.cosine_similarity(ours_moved, theirs_moved, dim=0) > 0.99
        assert torch.isclose(ours_moved.norm(), theirs_moved.norm(), rtol=0.05)


@pytest.mark.parametrize("name", NAMES)
def test_each_optimizer_in_the_bakeoff_lowers_the_loss(name):
    model = tiny()
    losses = train(model, make_optimizers(name, model, 1e-3), steps=20)
    assert losses[-1] < losses[0] - 0.2
