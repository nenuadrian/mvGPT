import argparse

import torch

from checkpoint import load_checkpoint
from mlx_bench import race
from mlx_model import from_torch as mlx_from_torch
from mlx_model import generate as mlx_generate
from model import pick_device
from mojo_model import from_torch as mojo_from_torch
from mojo_model import generate as mojo_generate

# Writing a story a token at a time on this Mac's GPU, three ways: PyTorch, MLX and our Mojo kernels. They take turns,
# with mlx_bench.py's race, so that whatever else the machine is doing slows them all alike.


def main():
    parser = argparse.ArgumentParser(description="PyTorch, MLX and our Mojo kernels, writing on this Mac's GPU.")
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    device = pick_device()
    torch_model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    torch_model.eval()
    mlx_model, _ = mlx_from_torch(args.checkpoint)
    mojo_model, _ = mojo_from_torch(args.checkpoint)
    ids = tokenizer.encode(args.prompt)

    def write_torch():
        out = torch_model.generate(torch.tensor([ids], device=device), args.tokens, top_k=1)
        torch.mps.synchronize()
        return out[0].tolist()

    # All three pick the likeliest token each time, so they should write the same story.
    best, out = race({
        "PyTorch": write_torch,
        "MLX": lambda: mlx_generate(mlx_model, ids, args.tokens, greedy=True),
        "Mojo": lambda: mojo_generate(mojo_model, ids, args.tokens, greedy=True),
    }, args.rounds)
    new = len(out["Mojo"]) - len(ids)
    print(f"writing {new} tokens, one at a time:")
    for name, seconds in best.items():
        print(f"  {name + ':':9}{seconds:5.2f} s, {new / seconds:5,.0f} tokens/s")
    print(f"  the same story: {out['PyTorch'] == out['MLX'] == out['Mojo']}")
    print("  " + tokenizer.decode(out["Mojo"]))


if __name__ == "__main__":
    main()
