import argparse
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Periwinkle for the training batches and amber for validation, on the series' near-black slate.
BATCH, VALID, SLATE, FRAME = "#6a82e6", "#c9822f", "#101215", "#3a3f47"


def read_run(folder):
    # The accumulator reads a run's event files, the same ones TensorBoard reads.
    events = EventAccumulator(folder)
    events.Reload()
    scalars = {}
    for tag in events.Tags()["scalars"]:
        # Each scalar is a list of events, with the wall time, the step and the value.
        points = events.Scalars(tag)
        scalars[tag] = ([p.step for p in points], [p.value for p in points])
    # A text summary is stored as a tensor of strings, one per evaluation.
    samples = [(e.step, e.tensor_proto.string_val[0].decode()) for e in events.Tensors("sample/text_summary")]
    return scalars, samples


def plot(name, scalars, path):
    plt.style.use("dark_background")
    # The learning rate gets a chart of its own, as it's on a scale of its own.
    panels = 2 if "lr" in scalars else 1
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 4), facecolor=SLATE, squeeze=False)
    loss = axes[0][0]
    loss.plot(*scalars["loss/batch"], color=BATCH, linewidth=2, label="a training batch")
    loss.plot(*scalars["loss/valid"], color=VALID, linewidth=2, marker="o", markersize=5, label="validation")
    loss.set(title="loss", xlabel="step")
    loss.legend(frameon=False)
    if panels == 2:
        lr = axes[0][1]
        lr.plot(*scalars["lr"], color=BATCH, linewidth=2)
        lr.set(title="learning rate", xlabel="step")
    for ax in axes[0]:
        ax.set_facecolor(SLATE)
        ax.spines[:].set_color(FRAME)
        ax.grid(alpha=0.15)
    fig.suptitle(name)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Read training runs' TensorBoard logs and plot them.")
    parser.add_argument("runs", nargs="+", help="run folders, such as runs/sft-20261015-101500")
    parser.add_argument("--out", default="results", help="the folder to write the charts to")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for folder in args.runs:
        name = os.path.basename(os.path.normpath(folder))
        scalars, samples = read_run(folder)
        print(name)
        for tag, (steps, values) in scalars.items():
            print(f"  {tag:<14} {len(steps):>4} points, last {values[-1]:.4g} at step {steps[-1]}")
        for step, sample in samples:
            print(f"  step {step:>5}: {sample[:80]}")
        path = os.path.join(args.out, f"{name}.png")
        plot(name, scalars, path)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
