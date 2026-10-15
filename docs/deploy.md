# Sharing and serving the model

A checkpoint in `checkpoints/` runs with this repo's code and nothing else. To let anyone use the
model, mvGPT goes three places: the [Hugging Face Hub](https://huggingface.co/nenuadrian/mvGPT),
where the weights are versioned and anyone can load them; a web app on a server, with Gradio in
Docker; and the browser, with Transformers.js, where it runs on the visitor's machine.

## The Hugging Face format

mvGPT has GPT-2's architecture: learned positions, LayerNorm before each sublayer, GELU, and a head
tied to the token embeddings. So [`hub.py`](https://github.com/nenuadrian/mvGPT/blob/main/hub.py)
saves it as transformers' `GPT2LMHeadModel`, by renaming its weights:

| mvGPT | GPT-2 in transformers |
| --- | --- |
| `embed_tokens`, `embed_positions` | `transformer.wte`, `transformer.wpe` |
| `blocks.i.attn_norm`, `blocks.i.mlp_norm` | `transformer.h.i.ln_1`, `transformer.h.i.ln_2` |
| `blocks.i.attn.q_proj`, `k_proj`, `v_proj` | `transformer.h.i.attn.c_attn`, the three joined, transposed |
| `blocks.i.attn.o_proj` | `transformer.h.i.attn.c_proj`, transposed |
| `blocks.i.mlp.up_proj`, `down_proj` | `transformer.h.i.mlp.c_fc`, `c_proj`, transposed |
| `norm` | `transformer.ln_f` |
| `lm_head` | `lm_head`, tied to `wte` |

GPT-2's layers store their weights as `(in, out)`, the transpose of `nn.Linear`'s `(out, in)`, and
its config sets `activation_function="gelu"`, the exact GELU we use, where GPT-2's default
approximates it with tanh.

The tokenizer becomes a `tokenizer.json`, the format of Hugging Face's tokenizers library: the
`Whitespace` pre-tokenizer splits text on `\w+|[^\w\s]+`, the regex our tokenizer uses, and a
`WordPiece` model with no `##` pieces looks each word up, or gives `[UNK]`, as ours does. Decoding
is the difference: the tokenizer drops whitespace, so `tokenizer.decode` puts spaces around some
punctuation (`Sam ' s`), and the app and the browser page join the tokens with `detokenize`, as
`tokenizer.py` does.

```bash
python hub.py checkpoints/sft/model.pt hf/sft
```

It saves the model as `model.safetensors`, which holds tensors and nothing else, where
`torch.save` writes a pickle, which can run code when it's loaded. Then it loads the folder back with
`AutoModelForCausalLM` and `AutoTokenizer` and checks it against our model: the token ids match on
46,192 tokens of held-out instructions, and the logits differ by about $1.5 \times 10^{-5}$.

## ONNX, and 8 bits for the browser

`hub.py` also exports the model to [ONNX](https://onnx.ai), a format for models that runtimes in
other languages can run, with optimum, as `onnx/model.onnx`. The export takes the KV cache as inputs
and returns it as outputs (`text-generation-with-past`). Then ONNX Runtime quantizes its weights to
8-bit integers, as `onnx/model_quantized.onnx`, for a browser to download:

| | Size | Loss on a held-out window |
| --- | --- | --- |
| PyTorch | | 3.3055 |
| `model.onnx` | 58.6 MB | 3.3055 |
| `model_quantized.onnx` | 15.0 MB | 3.3081 |

## On the Hub

```bash
hf auth login
python hub.py checkpoints/pretrain/model.pt hf/pretrain --push nenuadrian/mvGPT \
  --tag ep002-pretrain --message "Pretrained on TinyStories for ten minutes, in episode 2"
python hub.py checkpoints/sft/model.pt hf/sft --push nenuadrian/mvGPT \
  --tag ep003-sft --message "Fine-tuned on TinyStories-Instruct, in episode 3"
```

A model repo on the Hub is a git repo. Each push is a commit, with the folder's files and a model
card, `model_card.md` filled in as its `README.md`, and a tag names the commit after the episode.
`main` moves on with each push, while a tag stays where it was:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nenuadrian/mvGPT", revision="ep002-pretrain")
model = AutoModelForCausalLM.from_pretrained("nenuadrian/mvGPT", revision="ep002-pretrain")
```

## A web app with Gradio

[`app.py`](https://github.com/nenuadrian/mvGPT/blob/main/app.py) is a [Gradio](https://gradio.app)
app: pick the fine-tuned or the pretrained model, features, words and a summary, and it writes the
story for that instruction a token at a time. It loads each version from the Hub by its tag, the
first time it's picked, and generates with transformers' GPT-2, passing its KV cache,
`past_key_values`, from one step to the next.

```bash
python app.py    # http://127.0.0.1:7860
```

## On a server, with Docker

The [`Dockerfile`](https://github.com/nenuadrian/mvGPT/blob/main/Dockerfile) installs PyTorch's
CPU build and the app's pinned requirements, and runs the app on all interfaces, on port 7860.
[`deploy.sh`](https://github.com/nenuadrian/mvGPT/blob/main/deploy.sh) sends the committed files
to a server with `git archive` over SSH, builds the image there and swaps the running container for
a new one:

```bash
./deploy.sh debian@208.113.215.123    # http://208.113.215.123:7860
```

The container restarts with the server, and a volume keeps the models it downloaded from the Hub.
Docker publishes the port past a firewall such as ufw, so the app is public as soon as it runs. A
model this small writes a story in a few seconds on the server's two CPU cores, so it needs no GPU. The
app has no accounts or limits, and Gradio's queue runs one story at a time.

## In the browser, with Transformers.js

[`web/index.html`](https://github.com/nenuadrian/mvGPT/blob/main/web/index.html) is a single page.
On the first story, [Transformers.js](https://huggingface.co/docs/transformers.js) downloads
`onnx/model_quantized.onnx` and the tokenizer from the Hub, about 15 MB, which the browser then
caches, and ONNX Runtime runs the model in the tab, with WebAssembly. No server runs the model:
anything that serves the page will do.

```bash
python -m http.server -d web 8000    # http://localhost:8000
```

The series' website has a chat with the model built this way, which downloads it when you ask for a
story.
