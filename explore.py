import argparse

import model_explorer
import torch
from model_explorer import node_data_builder as nodes
from torch.profiler import record_function

from checkpoint import load_checkpoint
from profiler import profiled


class Labelled(torch.fx.Interpreter):
    # Runs the graph a node at a time, with each node labelled by its name, for the profiler.
    def run_node(self, node):
        with record_function(node.name):
            return super().run_node(node)


@torch.no_grad()
def node_times(program, x):
    graph = Labelled(program.module())
    graph.run(x)  # a first run, to warm up
    with profiled(x.device) as prof:
        graph.run(x)
    names = {node.name for node in program.graph.nodes if node.op == "call_function"}
    events = prof.key_averages()
    return {e.key: e.cpu_time_total / 1000 for e in events if e.key in names}


def main():
    parser = argparse.ArgumentParser(description="Show a model as a graph of its operations, in Model Explorer.")
    parser.add_argument("checkpoint")
    parser.add_argument("--times", action="store_true", help="colour each operation by how long it takes, on the CPU")
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
    if args.times:
        times = node_times(program, x)
        slowest = sorted(times.items(), key=lambda kv: -kv[1])[:5]
        print("slowest:", ", ".join(f"{name} {ms:.2f} ms" for name, ms in slowest))
        # From the slate for the quickest operations to coral for the slowest.
        data = nodes.GraphNodeData(
            results={name: nodes.NodeDataResult(value=round(ms, 3)) for name, ms in times.items()},
            gradient=[nodes.GradientItem(stop=0, bgColor="#191d22"), nodes.GradientItem(stop=1, bgColor="#ff8a65")],
        )
        config.add_node_data("milliseconds", nodes.ModelNodeData(graphsData={"graph": data}))
    model_explorer.visualize_from_config(config, port=args.port)


if __name__ == "__main__":
    main()
