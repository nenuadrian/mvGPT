---
license: mit
language: en
library_name: transformers
pipeline_tag: text-generation
datasets:
  - roneneldan/TinyStories
  - roneneldan/TinyStoriesInstruct
tags:
  - mvgpt
  - gpt2
  - tinystories
  - transformers.js
---

# mvGPT

**Minimum Viable GPT** is a small decoder-only transformer, 14.6 million parameters, written from
scratch in PyTorch for the YouTube series [Minimum Viable GPT](https://mvgpt.nenuadrian.com). It
writes short children's stories, in the style of
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). The code that trained it is at
<https://github.com/nenuadrian/mvGPT>, with a tag for each step of each episode.

This revision, `$tag`: $message.

mvGPT has GPT-2's architecture, so it's saved as a `GPT2LMHeadModel`: 6 blocks of 6 heads, an
embedding size of 384, 256 learned positions, GELU, and a head tied to the token embeddings. Its
vocabulary is 10,000 words and runs of punctuation, from its own word-level tokenizer, written here
as a `tokenizer.json`.

## Use it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("$repo", revision="$tag")
model = AutoModelForCausalLM.from_pretrained("$repo", revision="$tag")
ids = tokenizer("Features: Dialogue\nWords: dog, ball, happy\nStory:", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.8, top_k=50)
print(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

In a browser, with [Transformers.js](https://huggingface.co/docs/transformers.js), from the 8-bit
ONNX model in `onnx/`:

```js
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";

const writer = await pipeline("text-generation", "$repo", { dtype: "q8", revision: "$tag" });
const [out] = await writer("Once upon a time", { max_new_tokens: 100, do_sample: true, top_k: 50 });
```

## Revisions

Each episode that changes the model pushes a new commit here, tagged after the episode, such as
`ep002-pretrain` for the pretrained model and `ep003-sft` for the fine-tuned one. `revision=` loads
a tag, and `main` is the latest.

## Limitations

- It writes children's stories, and nothing else: no facts, arithmetic or chat.
- Its context window is 256 tokens.
- A word outside its vocabulary becomes `[UNK]`.
- The tokenizer drops whitespace, so `tokenizer.decode` puts spaces around some punctuation, as in
  `Sam ' s` and `Summary :`. `detokenize` in mvGPT's `tokenizer.py` joins them back.
