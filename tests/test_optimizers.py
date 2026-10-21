import pytest
import torch

from bakeoff import NAMES, make_optimizers
from model import GPT, GPTConfig
from optimizers import AdamW, Lion


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


@pytest.mark.parametrize("name", NAMES)
def test_each_optimizer_in_the_bakeoff_lowers_the_loss(name):
    model = tiny()
    losses = train(model, make_optimizers(name, model, 1e-3), steps=20)
    assert losses[-1] < losses[0] - 0.2
