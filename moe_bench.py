import argparse
import statistics
import time

import torch

from bench import sync
from model import GPT, MODERN, GPTConfig, pick_device
from pretrain import BATCH_SIZE

# A training step's time, the dense model's against a mixture of experts', on pretrain.py's batch. They take turns, a
# few steps each, so that whatever else the machine is doing slows them both down alike.


def main():
    parser = argparse.ArgumentParser(description="A training step's time, dense against a mixture of experts.")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5, help="steps in a row for each model, each round")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(pick_device())
    models = {}
    for name, experts in (("dense", 0), (f"{args.experts} experts", args.experts)):
        model = GPT(GPTConfig(vocab_size=10_000, n_expert=experts, **MODERN)).to(device)
        models[name] = (model, torch.optim.AdamW(model.parameters(), lr=1e-4))
    # Random tokens and targets: a step takes as long whatever the tokens are.
    x = torch.randint(10_000, (BATCH_SIZE, 256), device=device)
    y = torch.randint(10_000, (BATCH_SIZE, 256), device=device)

    def step(model, optimizer):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for model, optimizer in models.values():
        for _ in range(3):  # to warm up
            step(model, optimizer)
    times = {name: [] for name in models}
    for _ in range(args.rounds):
        for name, (model, optimizer) in models.items():
            sync(device)
            start = time.perf_counter()
            for _ in range(args.steps):
                step(model, optimizer)
            sync(device)
            times[name].append((time.perf_counter() - start) / args.steps * 1000)
    for name, (model, _) in models.items():
        ms = statistics.median(times[name])
        print(f"{name:>10}: {model.num_params() / 1e6:4.1f}M parameters, {ms:4.0f} ms a step, "
              f"{BATCH_SIZE * 256 / ms * 1000:,.0f} tokens/s")
    dense, moe = (statistics.median(t) for t in times.values())
    print(f"the mixture's step takes {moe / dense:.2f} times as long")


if __name__ == "__main__":
    main()
