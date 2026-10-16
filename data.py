import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

from tokenizer import EOT, BPETokenizer, WordTokenizer, build_vocab

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VOCAB_SIZE = 10_000
BPE_SIZE = 8_192  # 256 bytes, 7,934 merges, padding and end of text
HF = "https://huggingface.co/datasets/roneneldan"
DOWNLOADS = {
    "train.txt": f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "valid.txt": f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "instruct_train.txt": f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt",
    "instruct_valid.txt": f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt",
}
MAX_BYTES = 80 * 2**20  # the first 80 MB of each file is plenty for a model this size
# FineWeb-Edu's sample of ten billion GPT-2 tokens: 14 files of web pages, about 2 GB each.
FINEWEB = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
WEB_SHARDS = 14
WEB_BPE_SIZE = 32_768  # 256 bytes, 32,510 merges, padding and end of text: the ids still fit in 16 bits
WEB_MERGE_BYTES = 256 * 2**20  # the merges are learned from the first 256 MB of the first file
WEB_VALID_PAGES = 10_000  # held out: the first pages of the last file, which training never reads


def download(url, path, max_bytes=MAX_BYTES):
    with urllib.request.urlopen(url) as response:
        raw = response.read(max_bytes)
    text = raw.decode("utf-8", errors="ignore")
    text = text[: text.rfind(EOT) + len(EOT)]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{os.path.basename(path):<20}{len(raw) / 2**20:6.1f} MB {text.count(EOT):>9,} stories")


def encode(tokenizer, src, dst):
    with open(src, encoding="utf-8") as f:
        text = f.read()
    start = time.time()
    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    seconds = time.time() - start
    ids.tofile(dst)
    unknown = 0.0 if tokenizer.unk_id is None else (ids == tokenizer.unk_id).mean()
    print(f"{os.path.basename(dst):<20}{len(ids):>12,} tokens {unknown:8.2%} unknown "
          f"{len(text) / len(ids):5.2f} characters per token {seconds:6.1f}s")


def load_tokenizer(bpe=False, merges="merges.txt"):
    if bpe:
        with open(os.path.join(DATA_DIR, merges)) as f:
            return BPETokenizer([tuple(map(int, line.split())) for line in f])
    with open(os.path.join(DATA_DIR, "vocab.txt"), encoding="utf-8") as f:
        return WordTokenizer(f.read().split("\n"))


def build_bpe():
    import bpe

    # The merges are learned from the stories and the instructions, as the vocabulary was.
    text = ""
    for name in ("train.txt", "instruct_train.txt"):
        with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
            text += f.read().replace(EOT, " ")
    start = time.time()
    merges = bpe.train(text.encode("utf-8"), BPE_SIZE - 256 - 2)
    print(f"{len(merges):,} merges from {len(text) / 2**20:.0f} MB in {time.time() - start:.1f}s")
    with open(os.path.join(DATA_DIR, "merges.txt"), "w") as f:
        f.write("\n".join(f"{a} {b}" for a, b in merges))
    tokenizer = BPETokenizer(merges)
    for split in ("train", "valid"):
        encode(tokenizer, os.path.join(DATA_DIR, f"{split}.txt"), os.path.join(DATA_DIR, f"{split}-bpe.bin"))
    return tokenizer


def encode_part(text):
    return np.array(_tokenizer.encode(text.decode("utf-8")), dtype=np.uint16)


def start_worker(merges="merges.txt"):
    global _tokenizer
    _tokenizer = load_tokenizer(bpe=True, merges=merges)


def build_full():
    # All of TinyStories' training stories, 2.2 GB, for GPUs that can read them, with the merges learned on
    # the first 80 MB: the same tokens, and the same held-out stories, as the models trained on a laptop.
    tokenizer = load_tokenizer(bpe=True)
    valid = os.path.join(DATA_DIR, "valid.txt")
    if not os.path.exists(valid):
        download(DOWNLOADS["valid.txt"], valid)
    encode(tokenizer, valid, os.path.join(DATA_DIR, "valid-bpe.bin"))
    path = os.path.join(DATA_DIR, "train-full.txt")
    if not os.path.exists(path):
        download(DOWNLOADS["train.txt"], path, max_bytes=None)
    with open(path, "rb") as f:
        text = f.read()
    # Pieces of about 32 MB, each ending after a story, so they encode as the whole file would, a piece per core.
    end_of_text = EOT.encode()
    bounds = [0]
    while bounds[-1] < len(text):
        i = text.find(end_of_text, bounds[-1] + 32 * 2**20)
        bounds.append(len(text) if i < 0 else i + len(end_of_text))
    start = time.time()
    tokens = 0
    with Pool(os.cpu_count(), initializer=start_worker) as pool, \
            open(os.path.join(DATA_DIR, "train-full-bpe.bin"), "wb") as out:
        for ids in pool.imap(encode_part, (text[a:b] for a, b in zip(bounds, bounds[1:]))):
            ids.tofile(out)
            tokens += len(ids)
    print(f"train-full-bpe.bin {tokens:>14,} tokens {len(text) / tokens:5.2f} bytes per token "
          f"{time.time() - start:6.1f}s on {os.cpu_count()} cores")


def download_shard(i):
    path = os.path.join(DATA_DIR, "web", f"{i:03d}_00000.parquet")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Streamed to disk, as a file is about 2 GB, and renamed once it's whole.
        with urllib.request.urlopen(f"{FINEWEB}/{i:03d}_00000.parquet") as response, \
                open(path + ".part", "wb") as f:
            shutil.copyfileobj(response, f, 16 * 2**20)
        os.rename(path + ".part", path)
    return path


def web_pieces(path, pages=None, size=32 * 2**20):
    # A file's pages, as pieces of text of about 32 MB, in bytes, each page followed by end of text.
    import pyarrow.parquet as pq

    piece, length, seen = [], 0, 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=1024):
        for page in batch.column(0).to_pylist():
            piece.append(page + EOT)
            length += len(page)
            seen += 1
            if length >= size or seen == pages:
                yield "".join(piece).encode("utf-8")
                piece, length = [], 0
            if seen == pages:
                return
    if piece:
        yield "".join(piece).encode("utf-8")


def build_web(shards):
    # FineWeb-Edu: pages from the web that a classifier, trained on a big model's ratings, scored as
    # educational. Its own merges, as web pages are a different text from the stories.
    with ThreadPoolExecutor(4) as downloads:
        paths = list(downloads.map(download_shard, [*range(shards), WEB_SHARDS - 1]))
    merges_path = os.path.join(DATA_DIR, "web-merges.txt")
    if not os.path.exists(merges_path):
        import bpe

        text = []
        for piece in web_pieces(paths[0]):
            text.append(piece.replace(EOT.encode(), b" "))
            if sum(map(len, text)) >= WEB_MERGE_BYTES:
                break
        text = b"".join(text)
        start = time.time()
        merges = bpe.train(text, WEB_BPE_SIZE - 256 - 2)
        print(f"{len(merges):,} merges from {len(text) / 2**20:.0f} MB of web pages in {time.time() - start:.1f}s")
        with open(merges_path, "w") as f:
            f.write("\n".join(f"{a} {b}" for a, b in merges))
    tokenizer = load_tokenizer(bpe=True, merges="web-merges.txt")

    valid = os.path.join(DATA_DIR, "valid-web.txt")
    with open(valid, "wb") as f:
        for piece in web_pieces(paths[-1], pages=WEB_VALID_PAGES):
            f.write(piece)
    encode(tokenizer, valid, os.path.join(DATA_DIR, "valid-web.bin"))

    # Each file in pieces, a piece per core, appended to one file of tokens. A process gets a fresh tokenizer
    # after 4 pieces, as the encoder remembers each chunk it has seen, and the web has a lot of them.
    sizes = []

    def measured(pieces):
        for piece in pieces:
            sizes.append(len(piece))
            yield piece

    start = time.time()
    tokens = 0
    with Pool(os.cpu_count(), initializer=start_worker, initargs=("web-merges.txt",), maxtasksperchild=4) as pool, \
            open(os.path.join(DATA_DIR, "train-web.bin"), "wb") as out:
        for path in paths[:-1]:
            for ids in pool.imap(encode_part, measured(web_pieces(path))):
                ids.tofile(out)
                tokens += len(ids)
            print(f"{os.path.basename(path):<20}{tokens:>14,} tokens so far {time.time() - start:6.0f}s", flush=True)
    print(f"train-web.bin {tokens:>14,} tokens {sum(sizes) / tokens:5.2f} bytes per token "
          f"{time.time() - start:6.1f}s on {os.cpu_count()} cores")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if "--full" in sys.argv:
        build_full()
        return
    if "--web" in sys.argv:
        # How many of FineWeb-Edu's files to train on: python data.py --web 6
        build_web(int(sys.argv[sys.argv.index("--web") + 1]))
        return
    for name, url in DOWNLOADS.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            download(url, path)
    if "--bpe" in sys.argv:
        tokenizer = build_bpe()
        text = 'Once upon a time, a cat said "Hello!"'
        ids = tokenizer.encode(text)
        print()
        print(text)
        print([tokenizer.id_to_token[i] for i in ids])
        print(ids)
        print(tokenizer.decode(ids))
        return

    # The instructions count too: the vocabulary can't change once the model is trained.
    train = [os.path.join(DATA_DIR, n) for n in ("train.txt", "instruct_train.txt")]
    vocab, counts = build_vocab(train, VOCAB_SIZE)
    with open(os.path.join(DATA_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    print(f"vocabulary: {len(vocab):,} of {len(counts):,} distinct tokens")

    tokenizer = WordTokenizer(vocab)
    for split in ("train", "valid"):
        encode(tokenizer, os.path.join(DATA_DIR, f"{split}.txt"), os.path.join(DATA_DIR, f"{split}.bin"))

    # A sentence as the model sees it, and back.
    text = 'Once upon a time, a cat said "Hello!"'
    ids = tokenizer.encode(text)
    print()
    print(text)
    print([tokenizer.id_to_token[i] for i in ids])
    print(ids)
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
