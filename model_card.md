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
  - $arch_tag
  - tinystories
  - transformers.js
---

# mvGPT

**Minimum Viable GPT** is a small decoder-only transformer, written from scratch in PyTorch for the YouTube series [Minimum Viable GPT](https://mvgpt.nenuadrian.com). It
writes short children's stories, in the style of
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). The code that trained it is at
<https://github.com/nenuadrian/mvGPT>, with a tag for each step of each episode.

This revision, `$tag`: $message.

$architecture $vocabulary

## Use it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("$repo", revision="$tag")
model = AutoModelForCausalLM.from_pretrained("$repo", revision="$tag")
ids = tokenizer($prompt, return_tensors="pt").input_ids
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
`ep002-pretrain` for the pretrained model and `ep003-sft` for the fine-tuned one, and
`ep012-pretrain` for a model of 91 million parameters, pretrained on four GPUs. `revision=` loads
a tag, and `main` is the latest.

## Limitations

- It writes children's stories, and nothing else: no facts, arithmetic or chat.
- Its context window is $context tokens.
$limitations
