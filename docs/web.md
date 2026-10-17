# Pretraining on the web

The models so far learned from TinyStories, short stories written with the words a three-year-old
knows. This one, in GPT-2 small's shape, learns from web pages: FineWeb-Edu, for a night, on one RTX
4090 rented by the hour.

## FineWeb-Edu

[FineWeb](https://arxiv.org/abs/2406.17557) is 15 trillion tokens of English web pages from 96
Common Crawl snapshots, filtered and deduplicated by Hugging Face. FineWeb-Edu is the 1.3 trillion
tokens of it that a classifier scored 3 or more, out of 5, for educational value. Llama 3 70B scored
460,000 pages, and a linear layer on an embedding model learned to give its scores, then scored all
of FineWeb.

`data.py --web N` downloads N of the 14 files of FineWeb-Edu's 10-billion-token sample, about 2 GB
each, and the last file, whose first 10,000 pages are held out and never trained on.

- **Merges of their own.** 32,510 merges are learned from the first 256 MB of the first file, by
  the [C++ trainer](bpe.md), for a vocabulary of 32,768: 256 bytes, the merges, padding and end of
  text. The ids still fit in 16 bits. They're saved as `web-merges.txt`, next to the stories'
  `merges.txt`.
- **Encoded on every core**, a piece of about 32 MB per process, with each page followed by end of
  text. A process gets a fresh tokenizer after 4 pieces, as the encoder caches each chunk it sees.

On the first two million characters of the held-out pages:

| Merges | Characters per token |
| --- | --- |
| The stories' (8,192 tokens) | 3.23 |
| The web's (32,768 tokens) | 4.67 |

So a page takes 31% fewer tokens, and a window holds 45% more text.

```bash
python data.py --web 1        # on a laptop: the merges, the held-out pages, and one file (753 million tokens)
./remote.sh vast --web 5      # on a rented machine: five files, 3.76 billion tokens, with the laptop's merges
```

## A night's budget

The [Chinchilla paper](https://arxiv.org/abs/2203.15556) found that for a given amount of compute,
parameters and tokens should grow together, about 20 tokens per parameter. The model has
110,221,824 parameters: 12 blocks of 12 heads, an embedding size of 768, a context of 1,024 tokens,
rotary embeddings, and the web's vocabulary. That's 2.2 billion tokens at 20 per parameter. One RTX
4090 trains it at about 136,000 tokens a second, so a night of eight hours is about 3.9 billion.

## The new options

`pretrain_ddp.py` gets three:

- `--web` trains on `train-web.bin` with the web's merges, measures `valid-web.bin`, and writes
  from "Photosynthesis is" at each evaluation, into `checkpoints/pretrain-web`.
- `--accum N` adds up the gradients of N batches before each step, for a bigger batch than fits at
  once. Each loss is divided by N, and DDP's averaging across GPUs is turned off (`no_sync`) for all
  but the last batch.
- `--resume` carries on from the run's last evaluation. Each evaluation saves `resume.pt` next to
  the model: AdamW's state, the step, and the seconds trained, so the clock, and with it the
  learning rate's schedule, picks up where it was.

```bash
ssh vast 'cd mvGPT; nohup torchrun --standalone --nproc_per_node 1 pretrain_ddp.py 470 --web --compile --block 1024 --batch 24 --accum 4 > train.log 2>&1 < /dev/null & echo $! > train.pid'
ssh vast 'cd mvGPT && tail -f --pid $(cat train.pid) train.log'
```

## One night on one RTX 4090

Rented on [Vast.ai](https://cloud.vast.ai/?ref_id=746308) (a referral link), in Japan, at $0.459 an
hour: 470 minutes of training, 36,743 steps of 98,304 tokens, 3.61 billion tokens, 0.96 of a pass
over the five files, about 33 tokens per parameter. It trained at 136,000 tokens a second, 128,000
with the evaluations, and used 14.5 GB at most, so it fits a standard 24 GB 4090. The held-out loss
went from 10.61 to 3.11, falling fast in the first hour (3.56) and slowly after (3.27 at four hours),
and was still going down at the end. The machine came to $3.99, from renting it to destroying it.

From "Photosynthesis is", along the way:

| Step | Held-out loss | |
| --- | --- | --- |
| 250 | 5.43 | Photosynthesis is a type of to reduce the benefits of health. |
| 36,500 | 3.08 | Photosynthesis is the process of taking carbon from the air and producing oxygen from the water and converting it into sugar. |
| 36,743 | 3.11 | Photosynthesis is the process by which a star's photosynthetic pigments convert sunlight into food. |

It writes like the pages it read, fluently, with facts that often don't hold.

## HellaSwag

`hellaswag.py` scores a model on [HellaSwag](https://arxiv.org/abs/1905.07830)'s 10,042 validation
examples, from the copy on Hugging Face (`Rowan/hellaswag`): for each beginning of a scene, the
average loss of each of the four endings' tokens, and the lowest is the model's pick. Guessing gets
25%, people about 95%.

```bash
python hellaswag.py checkpoints/pretrain-web/model.pt
python hellaswag.py gpt2      # OpenAI's GPT-2 small, from transformers
```

| Model | Parameters | Right | Accuracy |
| --- | --- | --- | --- |
| Guessing | | | 25.00% |
| TinyStories, 30 minutes on four GPUs ([ep012](distributed.md)) | 91,347,456 | 2,746 | 27.35% |
| FineWeb-Edu, one night on one GPU | 110,221,824 | 2,957 | 29.45% |
| GPT-2 small | 124,439,808 | 2,967 | 29.55% |

Ten examples apart from GPT-2 small, well inside the half a percent a score moves by chance on a
test this size.

## Bits per character

`bits.py --web` measures the held-out web pages instead of the stories, so each model can be
measured on the other's text:

| Bits per character | On stories | On web pages |
| --- | --- | --- |
| TinyStories model | 0.386 | 2.400 |
| Web model | 0.946 | 0.956 |

The stories model is a specialist, and the web model a generalist, about as good on stories as on
web pages.

## What's real, and what's scaled down

The data, as it comes from Hugging Face, and the recipe, GPT-2 small's shape with AdamW and a cosine
schedule, are the real thing. The scale isn't: 3.6 billion tokens, where Llama 3 was pretrained on
over 15 trillion, and one GPU for a night. It's a pretrained model: it carries on text, knows few
facts, and doesn't follow instructions.
