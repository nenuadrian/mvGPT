import argparse
import os
import re
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import bpe
from data import DATA_DIR, load_tokenizer, web_pieces

# The BPE tokenizer on FineWeb-Edu's pages, encoded as data.py encodes them: its memory, where its time goes, what
# threads do for it, and how its time grows with the length of a chunk.
PAGES = os.path.join(DATA_DIR, "web", "000_00000.parquet")  # the first file, from data.py --web 1


def tokenizer(cache_size=None):
    tok = load_tokenizer(bpe=True, merges="web-merges.txt")
    if cache_size is not None:
        tok.encoder = bpe.Encoder(tok.merges, cache_size)
    return tok


def encode(tok, piece):
    return tok.encode_bytes(piece)  # as data.py's encode_part does


def peak_memory():
    # The most memory the process has held so far, in MB: ru_maxrss is in bytes on macOS, kilobytes on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def memory(args):
    tok = tokenizer()
    for i, piece in enumerate(islice(web_pieces(PAGES), args.pieces)):
        if args.fresh:
            tok = tokenizer()  # and the last one, with its cache, is freed
        start = time.perf_counter()
        encode(tok, piece)
        seconds = time.perf_counter() - start
        print(f"piece {i + 1}: {len(piece) / 2**20 / seconds:5.1f} MB/s, process peak {peak_memory():6,.0f} MB, "
              f"malloc holds {bpe.allocated() / 2**20:5,.0f} MB")


def speed(args):
    # The pieces on one thread, with a new tokenizer each time, three times over, and the quickest of the three.
    pieces = list(islice(web_pieces(PAGES), args.pieces))
    times = []
    for _ in range(3):
        tok = tokenizer()
        start = time.perf_counter()
        for piece in pieces:
            encode(tok, piece)
        times.append(time.perf_counter() - start)
    print(f"{len(pieces)} pieces, one thread: {sum(map(len, pieces)) / 2**20 / min(times):5.1f} MB/s, the best of 3")


def cache(args):
    # The pieces through a tokenizer whose cache holds at most so many chunks, one tokenizer for all the pieces. The
    # sizes take turns, three rounds of them, and each keeps its quickest, so that whatever else the machine is doing
    # slows them all alike.
    pieces = list(islice(web_pieces(PAGES), args.pieces))
    megabytes = sum(map(len, pieces)) / 2**20
    sizes = (0, 2**10, 2**12, 2**14, 2**16, 2**18, 2**20)
    best, held = {}, {}
    for _ in range(3):
        for size in sizes:
            tok = tokenizer(size)
            before = bpe.allocated()
            start = time.perf_counter()
            for piece in pieces:
                encode(tok, piece)
            best[size] = min(best.get(size, float("inf")), time.perf_counter() - start)
            held[size] = bpe.allocated() - before
    for size in sizes:
        print(f"{size:>9,} chunks: {megabytes / best[size]:5.1f} MB/s, the cache holds {held[size] / 2**20:5.1f} MB")


class Timed:
    """Stands in for the C++ encoder, and adds up the time its calls take."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.seconds = 0.0

    def __getattr__(self, name):
        method = getattr(self.encoder, name)

        def timed(*args):
            start = time.perf_counter()
            try:
                return method(*args)
            finally:
                self.seconds += time.perf_counter() - start

        return timed


def profile(args):
    # cProfile can't see into the C++ module's calls, so the encoder is timed by hand, and the rest is Python's.
    piece = next(web_pieces(PAGES))
    tok = tokenizer()
    tok.encoder = Timed(tok.encoder)
    start = time.perf_counter()
    encode(tok, piece)
    total = time.perf_counter() - start
    cpp = tok.encoder.seconds
    print(f"one piece of {len(piece) / 2**20:.0f} MB: {total:.2f}s")
    print(f"  in C++:    {cpp:.2f}s {cpp / total:6.1%}")
    print(f"  in Python: {total - cpp:.2f}s {(total - cpp) / total:6.1%}")


def threads(args):
    pieces = list(islice(web_pieces(PAGES), args.pieces))
    megabytes = sum(map(len, pieces)) / 2**20
    for n in (1, 2, 4, 8):
        tok = tokenizer()  # a new one each time, so that no run starts with what the last one cached
        start = time.perf_counter()
        with ThreadPoolExecutor(n) as pool:
            list(pool.map(lambda piece: encode(tok, piece), pieces))
        seconds = time.perf_counter() - start
        print(f"{n} threads: {seconds:5.2f}s {megabytes / seconds:6.1f} MB/s", flush=True)


def doubling(args):
    # A piece without its spaces and punctuation: its letters are one long chunk, as a language written without
    # spaces, like Chinese, gives.
    letters = re.sub(rb"[^A-Za-z0-9\x80-\xff]", b"", next(web_pieces(PAGES))).decode("utf-8", errors="ignore")
    last = None
    for kb in (1, 2, 4, 8, 16, 32, 64):
        text = letters[: kb * 1024]
        times = []
        for _ in range(3):
            tok = tokenizer()
            start = time.perf_counter()
            tok.encode(text)
            times.append(time.perf_counter() - start)
        ms = min(times) * 1000
        growth = "" if last is None else f"  {ms / last:4.1f}x the last"
        print(f"{kb:3} KB: {ms:8.1f} ms{growth}")
        last = ms


def main():
    parser = argparse.ArgumentParser(description="Measures the BPE tokenizer on FineWeb-Edu's pages.")
    parser.add_argument("what", choices=["memory", "speed", "cache", "profile", "threads", "doubling"])
    parser.add_argument("--pieces", type=int, default=8, help="pieces of 32 MB, for memory, speed, cache and threads")
    parser.add_argument("--fresh", action="store_true", help="memory: a new tokenizer for each piece")
    args = parser.parse_args()
    modes = {"memory": memory, "speed": speed, "cache": cache, "profile": profile, "threads": threads,
             "doubling": doubling}
    modes[args.what](args)


if __name__ == "__main__":
    main()
