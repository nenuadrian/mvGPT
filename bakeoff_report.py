import argparse
import glob
import json
import math
import os
import statistics

from bakeoff import NAMES, STEPS
from hellaswag import RESULTS


def runs():
    found = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "bakeoff", "*.json"))):
        with open(path) as f:
            found.append(json.load(f))
    return found


def sweep(found, name):
    # The held-out loss at the end of each full-length run with seed 0, by its peak learning rate.
    return {r["lr"]: r["valid"] for r in found if r["name"] == name and r["seed"] == 0 and r["steps"] == STEPS}


def short(lr):
    # 0.0012 as 1.2e-3, and 3.75e-05 as 3.75e-5.
    mantissa, exponent = f"{lr:.3e}".split("e")
    return f"{float(mantissa):g}e{int(exponent)}"


def best_lr(found, name):
    losses = sweep(found, name)
    return min(losses, key=losses.get)


def seeds(found, name, lr):
    # The full-length runs at a learning rate, one per seed.
    return [r for r in found if r["name"] == name and r["lr"] == lr and r["steps"] == STEPS]


def adamw_steps(found, loss):
    # The steps AdamW, at its best learning rate, needs to reach a loss: its runs of other lengths, read
    # between the two around the loss, in log steps. None outside the runs we have.
    lr = best_lr(found, "adamw")
    lengths = sorted((r["steps"], r["valid"]) for r in found if r["name"] == "adamw" and r["lr"] == lr and r["seed"] == 0)
    for (s0, l0), (s1, l1) in zip(lengths, lengths[1:]):
        if l1 <= loss <= l0:
            return math.exp(math.log(s0) + (l0 - loss) / (l0 - l1) * math.log(s1 / s0))
    return None


def print_sweep(found, names):
    sweeps = {name: sweep(found, name) for name in names}
    lrs = sorted({lr for losses in sweeps.values() for lr in losses})
    print(f"held-out loss after {STEPS:,} steps, by peak learning rate")
    print(f"{'lr':>8}" + "".join(f"{name:>9}" for name in names))
    for lr in lrs:
        print(f"{short(lr):>8}" + "".join(f"{sweeps[name][lr]:>9.3f}" if lr in sweeps[name] else " " * 9 for name in names))


def print_best(found, names):
    print(f"the best learning rate of each, {STEPS:,} steps")
    print(f"{'':8}{'lr':>8}{'seed 0':>8}{'3 seeds':>9}{'spread':>8}{'as AdamW':>10}"
          f"{'ms/step':>9}{'optimizer':>11}{'kept':>6}")
    for name in names:
        lr = best_lr(found, name)
        done = seeds(found, name, lr)
        first = next(r for r in done if r["seed"] == 0)
        # How far apart the seeds land: the highest held-out loss minus the lowest.
        spread = max(r["valid"] for r in done) - min(r["valid"] for r in done)
        # The steps AdamW needs to reach seed 0's loss. With seed 0, all the optimizers start from one model
        # and read the batches in one order, so AdamW's own row is its 1,000 steps.
        steps = adamw_steps(found, first["valid"])
        # A step's time doesn't depend on the learning rate, so it's the median over all the optimizer's
        # full-length runs, as other work on the machine slows a run down now and then.
        timed = [r for r in found if r["name"] == name and r["steps"] == STEPS]
        ms, in_optimizer = (statistics.median(r[key] for r in timed) for key in ("ms_per_step", "ms_optimizer"))
        print(f"{name:<8}{short(lr):>8}{first['valid']:>8.3f}{statistics.mean(r['valid'] for r in done):>9.3f}{spread:>8.3f}"
              f"{f'{steps:,.0f}' if steps else '-':>10}{ms:>9.0f}{in_optimizer:>11.0f}{first['state'] / first['params']:>6.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The bake-off's results, from results/bakeoff.")
    # The optimizers to show, all of them when none are named.
    parser.add_argument("names", nargs="*", choices=NAMES)
    parser.add_argument("--best", action="store_true", help="the best learning rate of each, with seeds and AdamW's steps")
    args = parser.parse_args()
    (print_best if args.best else print_sweep)(runs(), args.names or NAMES)
