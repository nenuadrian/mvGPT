# Serving many at once

The [Gradio app](deploy.md) writes one story at a time, which is fine for a visitor trying the model.
A game is a crowd. [nanoMMO](https://nanommo.nenuadrian.com), the MMO of this series' sibling
series, has a storyteller that players can ask for a story, and many of them can ask at once.
[`server.py`](https://github.com/nenuadrian/mvGPT/blob/main/server.py) serves the model as an API,
and writes the requests that arrive together as a batch.

## A completions API

The server answers `POST /v1/completions` in the shape of
[OpenAI's completions API](https://platform.openai.com/docs/api-reference/completions), which most
clients and model servers speak, [vLLM](https://docs.vllm.ai)'s among them:

```bash
python server.py    # http://127.0.0.1:8000
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt": "Words: dragon, castle, brave\nStory:", "max_tokens": 100}'
```

```json
{
  "id": "cmpl-db90da905a0548d6b50f4667e0e21418",
  "object": "text_completion",
  "created": 1792087112,
  "model": "nenuadrian/mvGPT@ep003-sft",
  "choices": [{"index": 0, "text": "Once upon a time, there was a brave knight named George. George lived in a small castle. ...",
               "finish_reason": "length"}],
  "usage": {"prompt_tokens": 9, "completion_tokens": 100, "total_tokens": 109}
}
```

A request has a `prompt`, and can set `max_tokens` (100), `temperature` (0.8) and `top_k` (50), an
extra field vLLM takes too. `finish_reason` is `stop` when the model wrote the end of text, and
`length` when it reached `max_tokens`. The prompt and what's written after it have to fit in the
context window: a longer prompt gets a 400, and `max_tokens` is cut to the room left. OpenAI's own
Python client works with it, pointed at the server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
story = client.completions.create(model="mvgpt", prompt="Words: dragon, castle, brave\nStory:",
                                  max_tokens=100, extra_body={"top_k": 50})
```

`GET /health` says which model is served. It loads from the Hub by its tag, `ep003-sft` unless
`MVGPT_REVISION` names another, such as `ep012-pretrain`, which carries on the beginning of a story
instead of following an instruction. The word-level tokenizer's `decode` puts spaces around
punctuation (`Sam ' s`), so the server joins its tokens with `detokenize`, as the app does.

## How it's built

Python's `ThreadingHTTPServer` gives each connection a thread of its own. The thread reads the
request, puts it on a queue and waits on an event. The model runs on one thread, which takes
requests from the queue, writes them, and sets their events, and each connection's thread then
sends its answer. The model never runs on two threads at once, and the connections never wait on
each other's sockets. The server lets 128 connections wait to be accepted: with the default, 5, a
crowd arriving at once gets some of its connections reset.

## Measured: one at a time

[`load.py`](https://github.com/nenuadrian/mvGPT/blob/main/load.py) plays a crowd. Each client is a
player asking the storyteller for a story with three words from nanoMMO's world, ones the model's
vocabulary knows (sword, gate, skeleton, dragon...): it sends a request for 100 tokens, waits for the
story, and asks again, four times. It runs 1, 4, 16, then 64 clients at once:

```bash
python load.py --out results/load.json
```

On an Apple M4 Pro's CPU, writing one request at a time, the server writes about the same number of
tokens a second whatever the crowd, so a player's wait grows with the queue in front of them:

| Clients | Tokens a second | Median wait | 95th percentile |
| --- | --- | --- | --- |
| 1 | 674 | 149 ms | 149 ms |
| 4 | 682 | 584 ms | 590 ms |
| 16 | 686 | 2,323 ms | 2,337 ms |
| 64 | 671 | 9,370 ms | 9,603 ms |

## Batching

A step of the model reads all its weights, whatever the batch. For a model this small, on a CPU, a
step for 16 rows costs little more than a step for one, so writing requests together writes more
tokens a second. The model's thread takes the first request in the queue, gives others 5 ms to join
it (`--wait`), up to 16 (`--batch`), and writes them together. Without the wait, the first of a
crowd would be written alone, with the rest queued behind it.

The prompts are of different lengths, so they're padded on the left, with the padding token, to
the longest. As tokens (the tokenizer drops the line breaks):

```text
[PAD] [PAD] [PAD] [PAD] Words : dragon , castle Story :
[PAD] [PAD] Words : sword , gate , brave Story :
Words : skeleton , river , gold , king Story :
```

Each prompt ends in the last column, so every row's next token goes after it, and each step adds a
column. Two things keep the padding from changing what's written:

- **The attention mask**, a 1 for each token and a 0 for each pad, tells attention to leave the pads
  out, so no token attends to one.
- **The positions**, `mask.cumsum(1) - 1`, count each row's own tokens, not the pads in front of
  them, so a prompt's first token is at position 0 however much padding it has.

Each request keeps its own temperature, as a column of them that divides the logits, and its own
top k: its k-th largest logit, read from the largest k in the batch. A request that's finished, at
the end of text or at its `max_tokens`, keeps its row until the whole batch is done, and what the
row writes is dropped. [`tests/test_server.py`](https://github.com/nenuadrian/mvGPT/blob/main/tests/test_server.py)
checks that a batch of prompts of different lengths writes what each writes alone, with a tiny GPT-2
that always picks its likeliest token. Without the mask, or without the positions, it fails.

On the same CPU, with batches of up to 16 and up to 64:

| Clients | One at a time | Batches of 16 | Batches of 64 |
| --- | --- | --- | --- |
| 1 | 674 tokens/s, 149 ms | 608 tokens/s, 162 ms | 611 tokens/s, 162 ms |
| 4 | 682 tokens/s, 584 ms | 1,047 tokens/s, 381 ms | 1,053 tokens/s, 381 ms |
| 16 | 686 tokens/s, 2,323 ms | 2,977 tokens/s, 522 ms | 2,974 tokens/s, 520 ms |
| 64 | 671 tokens/s, 9,370 ms | 3,085 tokens/s, 2,040 ms | 3,511 tokens/s, 1,942 ms |

The times are median waits. A lone player waits about 13 ms more, the 5 ms wait and a batch's extra
work, and 16 players wait less than a quarter as long. Without the wait, in batches of 64, 4 clients
got 804 tokens a second and 16 got 2,343. At 64 clients in batches of 64, runs vary between about
3,500 and 4,300 tokens a second.

Batches of 16 are the default because a batch's KV cache grows with its rows. For `ep003-sft`, a
token's keys and values take 18 KB over the six blocks, so 16 full rows of 256 tokens hold 75 MB.
`ep012-pretrain` has twelve blocks of 768 channels and a window of 512 tokens: 72 KB a token, and
2.4 GB for 64 full rows.

## In Docker

The [`Dockerfile`](https://github.com/nenuadrian/mvGPT/blob/main/Dockerfile) has a stage for the
server beside the app's, with PyTorch's CPU build and transformers. It listens on port 8000, the
port vLLM's server uses too:

```bash
docker build --target server -t mvgpt-server .
docker run -p 8000:8000 -v mvgpt-hub:/root/.cache/huggingface mvgpt-server
```

`docker build` without a `--target` still makes the app, which `deploy.sh` builds. On the Mac, in
Docker, the server writes 184 tokens a second for one client and about 975 for 16 or 64 with
batches of 16: PyTorch's Linux build for ARM, in Docker's virtual machine, is slower than its macOS build, and
batching still writes five times as much.

## What these batches don't do

A batch here starts together and ends together. Rows that finish early ride along until the last
one is done, and a request that arrives just after a batch starts waits for the whole of it. Servers
built for this schedule each step instead, which is called continuous batching: a finished row
leaves at once, and a waiting request takes its place. [vLLM](https://docs.vllm.ai) does that, and
keeps the KV cache in fixed-size blocks, PagedAttention, so that sequences of different lengths
share memory without padding, which matters most on a GPU.

vLLM is the alternative to `server.py`, and it runs without a GPU too. It serves both kinds of model
on the Hub, `ep003-sft` as a GPT-2 and `ep012-pretrain` as a GPT-NeoX, and it has images for the CPU,
`vllm/vllm-openai-cpu`, for x86 (AVX-512, or AVX2 with fewer features) and ARM. It speaks the same
API on the same port, so a client written for one works with the other. It decodes with the
tokenizer's `decode`, so `ep003-sft`'s stories come back as `Sam ' s`, and `ep012-pretrain`'s
byte-level BPE as written.
