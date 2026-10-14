"""Downloads TinyStories, builds the vocabulary and stores the stories as token ids."""
import os
import urllib.request

import numpy as np

from tokenizer import EOT, WordTokenizer, build_vocab

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VOCAB_SIZE = 10_000
HF = "https://huggingface.co/datasets/roneneldan"
DOWNLOADS = {
    "train.txt": f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "valid.txt": f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "instruct_train.txt": f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt",
    "instruct_valid.txt": f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt",
}
MAX_BYTES = 80 * 2**20  # the first 80 MB of each file is plenty for a model this size


def download(url, path):
    with urllib.request.urlopen(url) as response:
        raw = response.read(MAX_BYTES)
    text = raw.decode("utf-8", errors="ignore")
    text = text[: text.rfind(EOT) + len(EOT)]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{os.path.basename(path):<20}{len(raw) / 2**20:6.1f} MB {text.count(EOT):>9,} stories")


def encode(tokenizer, src, dst):
    with open(src, encoding="utf-8") as f:
        ids = np.array(tokenizer.encode(f.read()), dtype=np.uint16)
    ids.tofile(dst)
    unknown = (ids == tokenizer.unk_id).mean()
    print(f"{os.path.basename(dst):<20}{len(ids):>12,} tokens {unknown:8.2%} unknown")


def load_tokenizer():
    with open(os.path.join(DATA_DIR, "vocab.txt"), encoding="utf-8") as f:
        return WordTokenizer(f.read().split("\n"))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in DOWNLOADS.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            download(url, path)

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
