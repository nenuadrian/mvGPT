import argparse
import glob
import json
import os

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


def main(names):
    found = runs()
    sweeps = {name: sweep(found, name) for name in names}
    lrs = sorted({lr for losses in sweeps.values() for lr in losses})
    print(f"held-out loss after {STEPS:,} steps, by peak learning rate")
    print(f"{'lr':>8}" + "".join(f"{name:>9}" for name in names))
    for lr in lrs:
        print(f"{short(lr):>8}" + "".join(f"{sweeps[name][lr]:>9.3f}" if lr in sweeps[name] else " " * 9 for name in names))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The bake-off's results, from results/bakeoff.")
    # The optimizers to show, all of them when none are named.
    parser.add_argument("names", nargs="*", choices=NAMES)
    main(parser.parse_args().names or NAMES)
