# The tokenizer, measured and fixed

[Episode 11](bpe.md)'s C++ BPE tokenizer, as [episode 19](web.md) runs it over FineWeb-Edu, measured with
`bpe_measure.py` before anything changes, then fixed a step at a time, each step measured again. The numbers are from
the episode's recordings, on an Apple M4 Pro (8 performance and 4 efficiency cores), with the web merges (32,768
tokens) and pieces of 32 MB, as `data.py` encodes them; they move from run to run, and from machine to machine.

```bash
python setup.py build_ext --inplace
python bpe_measure.py memory --pieces 6          # each piece's speed, the process's peak, and what malloc holds
python bpe_measure.py memory --pieces 6 --fresh  # with a new tokenizer for each piece
python bpe_measure.py speed                      # eight pieces on one thread, the best of three
python bpe_measure.py profile                    # a piece's time, in the C++ and in the Python around it
python bpe_measure.py threads                    # eight pieces on 1, 2, 4 and 8 threads sharing a tokenizer
python bpe_measure.py doubling                   # one chunk of letters, from 1 KB to 64 KB
python bpe_measure.py cache --pieces 4           # caches of 0 to a million chunks
python bpe_measure.py lengths                    # chunks of 4 to 256 bytes, scanned and merged off a heap
./tsan.sh python bpe_measure.py threads --pieces 2   # the threads under ThreadSanitizer
python -m pytest tests/test_bpe.py
```

## Measured first

**Memory.** The process's peak (`ru_maxrss`) climbs with each piece, with one tokenizer or a new one for each piece,
and even when the pieces are decoded and nothing else. What malloc holds (`bpe.allocated()`: `malloc_zone_statistics`
on macOS, `mallinfo2` on Linux) shows the cache's own growth, which is much smaller:

| After 6 pieces | Process peak | Malloc holds |
| --- | --- | --- |
| One tokenizer | 924 → 2,066 MB | 142 → 197 MB |
| A new tokenizer each piece | 926 → 1,808 MB | 142 → 148 MB |
| Decoding alone | 574 → 1,661 MB | 120 → 126 MB |

A piece was a Python string twice on its way to the encoder: `web_pieces` joined the pages as one `str` before
encoding it, and `encode_part` decoded it again. Web pages have emoji, so such a string takes four bytes a character.
Python frees them and malloc counts them as given back, but macOS keeps counting them against the process.

**Time.** Of a piece's 0.74 s, 62% was in the C++ and 38% in the Python around it: the regular expression that split
out the special tokens, each page's text back to bytes, a Python list of the ids, and NumPy's copy of the list.
cProfile can't see inside pybind11's calls, so `profile` times the encoder by hand. One thread encoded about 43 MB/s.

**Threads.** More threads were slower, from 39.4 MB/s on one to 20.2 on eight, as the encoder held the GIL from start
to end and the threads took turns.

**Long chunks.** Each doubling of a chunk's length took about 2.8 times as long: 64 KB of letters without spaces took
370 ms. FineWeb-Edu's longest chunk, in a whole file of 726,000 pages, is 899 bytes, so it doesn't reach this, but
text without spaces does.

## The fixes

**An LRU cache.** `LruCache` keeps at most `cache_size` chunks (65,536 by default): a `std::list` in the order they
were used, the most recent first, and an `unordered_map` from a view of each chunk to its place in the list. A hit
moves the entry to the front with `splice`; a new chunk goes in at the front, and the one at the back makes room.

| Cache, chunks | MB/s | The cache holds |
| --- | --- | --- |
| none | 24.2 | 0.0 MB |
| 1K | 24.3 | 0.1 MB |
| 4K | 27.5 | 0.6 MB |
| 16K | 34.9 | 2.5 MB |
| 64K | 42.4 | 10.1 MB |
| 256K | 43.9 | 41.2 MB |
| 1M | 44.5 | 77.7 MB |

**Bytes in, NumPy out.** `Encoder` takes the special tokens' bytes and ids and finds them in the text, searching it
once; `encode` reads the bytes where Python keeps them and returns a NumPy array, and `BPETokenizer.encode_bytes`
gives `data.py` uint16 ids without a `str` or a list on the way. `web_pieces` joins each page's bytes. A test checks
that the held-out web pages give the 10,252,924 ids `data.py` wrote in episode 19. After it, the C++ had 99.9% of a
piece's 0.40 s, one thread did about 80 MB/s, and the process's peak went from 331 to 569 MB in 6 pieces.

**The GIL, and a data race.** `encode` releases the GIL (`py::gil_scoped_release`) while the C++ works. Two threads
sharing one tokenizer then crashed it, and `tsan.sh`, which builds the module with `SANITIZE=thread` and loads
ThreadSanitizer's runtime into Python, reported the race: `LruCache::Find` reading the cache's map while
`LruCache::Add` grew it on another thread.

**A cache per call.** `Encode` makes its own `LruCache` and is `const`, so threads share the encoder and nothing they
write. `data.py` encodes on a `ThreadPoolExecutor` of a thread per core sharing one tokenizer, with `in_order` keeping
two pieces per thread read ahead, instead of processes replaced every four pieces.

| Threads | Step 1, MB/s | Step 5, MB/s |
| --- | --- | --- |
| 1 | 39.4 | 82.4 |
| 2 | 23.9 | 152.5 |
| 4 | 20.9 | 258.8 |
| 8 | 20.2 | 438.5 |

**A heap for long chunks.** `EncodeLong` keeps a chunk's tokens as a linked list over arrays and a min-heap of the
neighbouring pairs that have a merge, by rank and then position; a stale pair is skipped when it comes off, and a
merge pushes the two pairs it makes. Each doubling of a chunk now takes about 2.3 times as long, and 64 KB takes 8.2
ms. Tests compare 500 random texts, and a chunk of 3,000 letters, with `bpe_reference.py`.

**Short chunks scanned, long ones off the heap.** The heap costs more to set up than a scan of a few pairs, and most
chunks are a few bytes, so `Encoder` takes `heap_from` (16): shorter chunks use the old loop.

| Chunk | Scanning | The heap |
| --- | --- | --- |
| 4 bytes | 0.10 µs | 0.24 µs |
| 8 bytes | 0.32 µs | 0.53 µs |
| 16 bytes | 1.19 µs | 1.13 µs |
| 32 bytes | 3.83 µs | 2.29 µs |
| 64 bytes | 10.48 µs | 4.64 µs |
| 128 bytes | 26.87 µs | 9.47 µs |
| 256 bytes | 68.69 µs | 19.34 µs |

| One thread, eight pieces | MB/s |
| --- | --- |
| Step 1, as it was | 43.0 |
| Step 3, bytes in, NumPy out | 80.0 |
| Step 6, the heap for all chunks | 72.1 |
| Step 7, the heap from 16 bytes | 82.3 |
