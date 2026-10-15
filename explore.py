import argparse

import model_explorer
import torch

from checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Show a model as a graph of its operations, in Model Explorer.")
    parser.add_argument("checkpoint")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    model, _, _ = load_checkpoint(args.checkpoint)
    model.eval()
    x = torch.randint(model.config.vocab_size, (1, model.config.block_size))
    # torch.export runs the model on x, and records each operation it does, with its shapes, as a graph.
    program = torch.export.export(model, (x,))
    calls = [node for node in program.graph.nodes if node.op == "call_function"]
    print(f"{len(calls)} operations for {x.size(1)} tokens")

    config = model_explorer.config().add_model_from_pytorch("mvGPT", exported_program=program)
    model_explorer.visualize_from_config(config, port=args.port)


if __name__ == "__main__":
    main()
