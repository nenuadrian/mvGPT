import torch
from torch.nn import functional as F


class AdamW(torch.optim.Optimizer):
    # Adam (Kingma and Ba, 2014) with decoupled weight decay (Loshchilov and Hutter, 2017): each weight
    # moves by the running average of its gradient, divided by the square root of the running average of
    # the gradient squared.
    def __init__(self, params, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    # Two numbers per weight: the momentum, m, and the average of the gradient squared, v.
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["step"] += 1
                m, v, t = state["m"], state["v"], state["step"]
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).add_(p.grad.square(), alpha=1 - beta2)
                # Both start at zero, so the first averages are too small: dividing by 1 - beta^t corrects them.
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                # Decoupled weight decay: the weights shrink a little each step, apart from the update.
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.sub_(group["lr"] * m_hat / (v_hat.sqrt() + group["eps"]))


class Lion(torch.optim.Optimizer):
    # Lion (Chen and others, 2023), found by a search over programs: each weight moves by the learning rate,
    # up or down, by the sign of a blend of the momentum and the gradient.
    def __init__(self, params, lr=1e-4, betas=(0.95, 0.98), weight_decay=1.0):
        # The betas its paper used for language models; the search had found 0.9 and 0.99.
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    # One number per weight: the momentum.
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                update = (beta1 * m + (1 - beta1) * p.grad).sign()
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.sub_(group["lr"] * update)
                # The momentum is updated after the step, with a slower beta than the blend's.
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)


class Sophia(torch.optim.Optimizer):
    # Sophia (Liu and others, 2023), SophiaG: the momentum divided by an estimate of the curvature, the loss's
    # second derivative for each weight, with each weight's step clipped to the learning rate.
    def __init__(self, params, lr=6e-4, betas=(0.965, 0.99), rho=0.05, weight_decay=0.2, tokens=12_288):
        # tokens: a batch's tokens, which the estimate of the curvature is scaled by.
        super().__init__(params, dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, tokens=tokens))

    def state_of(self, p):
        state = self.state[p]
        if not state:
            # Two numbers per weight: the momentum, m, and the curvature's running average, h.
            state.update(m=torch.zeros_like(p), h=torch.zeros_like(p))
        return state

    @torch.no_grad()
    def update_curvature(self):
        # After a backward pass on labels the model drew from its own predictions (see curvature, below).
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if p.grad is not None:
                    self.state_of(p)["h"].mul_(beta2).add_(p.grad.square(), alpha=1 - beta2)

    @torch.no_grad()
    def clipped(self):
        # The share of the weights whose step is clipped to the learning rate.
        hit = total = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    state = self.state[p]
                    hit += (state["m"].abs() >= group["rho"] * group["tokens"] * state["h"]).sum().item()
                    total += p.numel()
        return hit / max(total, 1)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state_of(p)
                m, h = state["m"], state["h"]
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                # The momentum over the curvature, a Newton step, clipped at 1: no weight moves more than lr.
                ratio = (m.abs() / (group["rho"] * group["tokens"] * h + 1e-15)).clamp(max=1)
                p.sub_(group["lr"] * m.sign() * ratio)


def curvature(model, x, optimizer, generator):
    # SophiaG's estimate, Gauss-Newton-Bartlett: labels drawn from the model's own predictions, and the
    # gradient of the loss on them, squared, averages to the curvature.
    logits, _ = model(x)
    logits = logits.view(-1, logits.size(-1))
    y = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator).view(-1)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(logits, y).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.update_curvature()
    optimizer.zero_grad(set_to_none=True)

