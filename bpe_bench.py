import os
import time

import bpe
import bpe_reference as reference
from data import DATA_DIR

MERGES = 500
SIZES = (100_000, 1_000_000)  # bytes of the training stories


def timed(f, *args):
    start = time.perf_counter()
    result = f(*args)
    return result, time.perf_counter() - start


def main():
    with open(os.path.join(DATA_DIR, "train.txt"), "rb") as f:
        text = f.read(max(SIZES))
    print(f"{'':<30}{'Python':>10}{'C++':>10}{'faster':>10}")
    for size in SIZES:
        sample = text[:size]
        merges, python = timed(reference.train, sample, MERGES)
        cpp_merges, cpp = timed(bpe.train, sample, MERGES)
        # Both give the same merges, or the times wouldn't mean much.
        assert [tuple(m) for m in cpp_merges] == merges
        print(f"{f'train, {size / 1e6:g} MB, {MERGES} merges':<30}{python:>9.2f}s{cpp:>9.3f}s{python / cpp:>9.0f}x")
    for size in SIZES:
        sample = text[:size]
        ids, python = timed(reference.encode, sample, merges)
        cpp_ids, cpp = timed(bpe.Encoder(merges).encode, sample)
        assert cpp_ids == ids
        print(f"{f'encode, {size / 1e6:g} MB':<30}{python:>9.2f}s{cpp:>9.3f}s{python / cpp:>9.0f}x")


if __name__ == "__main__":
    main()
