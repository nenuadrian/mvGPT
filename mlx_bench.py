import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch

from checkpoint import load_checkpoint
from mlx_model import GPT as MlxGPT
from mlx_model import from_torch, generate
from model import GPT, GPTConfig, pick_device
from pretrain import BATCH_SIZE

# MLX against PyTorch on this Mac's GPU: writing a story a token at a time, and training steps on pretrain.py's batch.
# The contenders take turns, as in moe_bench.py, so that whatever else the machine is doing slows them both alike.


def race(contenders, rounds):
    """Each contender once to warm up, then `rounds` turns each, in turn: the fastest time of each, and its output."""
    for run in contenders.values():
        run()
    best, out = {}, {}
    for _ in range(rounds):
        for name, run in contenders.items():
            start = time.perf_counter()
            out[name] = run()
            seconds = time.perf_counter() - start
            best[name] = min(best.get(name, seconds), seconds)
    return best, out


def writing(path, prompt, tokens, rounds):
    device = pick_device()
    torch_model, tokenizer, _ = load_checkpoint(path, device)
    torch_model.eval()
    mlx_model, _ = from_torch(path)
    ids = tokenizer.encode(prompt)

    def write_torch():
        out = torch_model.generate(torch.tensor([ids], device=device), tokens, top_k=1)
        # The GPU works in the background: wait for it, or the clock stops before the work does.
        torch.mps.synchronize()
        return out[0].tolist()

    # Both pick the likeliest token each time, so they should write the same story.
    best, out = race({"PyTorch": write_torch, "MLX": lambda: generate(mlx_model, ids, tokens, greedy=True)}, rounds)
    new = len(out["MLX"]) - len(ids)
    print(f"writing {new} tokens, one at a time:")
    for name, seconds in best.items():
        print(f"  {name + ':':9}{seconds:5.2f} s, {new / seconds:5,.0f} tokens/s")
    print(f"  the same story: {out['PyTorch'] == out['MLX']}")
    print("  " + tokenizer.decode(out["MLX"]))


def training(steps, rounds):
    # Episode two's model, and a batch of pretrain.py's size, of random tokens: the time of a step doesn't depend on
    # which tokens they are. Without dropout, as MLX's fused attention has none, so both do the same work.
    config = dict(vocab_size=10_000, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=True)
    x = np.random.default_rng(0).integers(0, 10_000, (BATCH_SIZE, 257))
    device = pick_device()

    torch_model = GPT(GPTConfig(**config)).to(device)
    torch_opt = torch.optim.AdamW(torch_model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    tx, ty = torch.from_numpy(x[:, :-1]).to(device), torch.from_numpy(x[:, 1:]).to(device)

    def torch_steps():
        for _ in range(steps):
            _, loss = torch_model(tx, ty)
            torch_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0)
            torch_opt.step()
        torch.mps.synchronize()

    model = MlxGPT(config)
    model.train()
    opt = optim.AdamW(6e-4, betas=[0.9, 0.95], weight_decay=0.1)
    mx_x, mx_y = mx.array(x[:, :-1].astype(np.int32)), mx.array(x[:, 1:].astype(np.int32))
    loss_and_grad = nn.value_and_grad(model, lambda m, a, b: nn.losses.cross_entropy(m(a), b, reduction="mean"))

    def step():
        loss, grads = loss_and_grad(model, mx_x, mx_y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    state = [model.state, opt.state, mx.random.state]
    compiled = mx.compile(step, inputs=state, outputs=state)

    def mlx_steps(fn):
        for _ in range(steps):
            fn()
            # MLX builds the work up lazily: eval runs it, step by step, as a training loop would.
            mx.eval(state)

    best, _ = race({"PyTorch": torch_steps, "MLX": lambda: mlx_steps(step),
                    "MLX, compiled": lambda: mlx_steps(compiled)}, rounds)
    print(f"training, {BATCH_SIZE} x 256 tokens a step:")
    for name, seconds in best.items():
        print(f"  {name + ':':15}{seconds / steps * 1000:5.0f} ms a step, {steps * BATCH_SIZE * 256 / seconds:7,.0f} tokens/s")


def main():
    parser = argparse.ArgumentParser(description="MLX against PyTorch on this Mac's GPU.")
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--steps", type=int, default=5, help="training steps in a row, each turn")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    writing(args.checkpoint, args.prompt, args.tokens, args.rounds)
    training(args.steps, args.rounds)


if __name__ == "__main__":
    main()
