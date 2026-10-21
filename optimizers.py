import torch


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

