import argparse
import json
import os
from string import Template

import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from huggingface_hub import HfApi
from optimum.exporters.onnx import main_export
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE, WordPiece
from torch.nn import functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, GPTNeoXConfig,
                          GPTNeoXForCausalLM, PreTrainedTokenizerFast)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from checkpoint import load_checkpoint
from data import DATA_DIR
from tokenizer import EOT, PAD, SPECIALS, UNK, BPETokenizer


def to_gpt2(model, tokenizer):
    c = model.config
    config = GPT2Config(
        vocab_size=c.vocab_size, n_positions=c.block_size, n_embd=c.n_embd, n_layer=c.n_layer, n_head=c.n_head,
        # Our GELU is the exact one, with erf, where GPT-2's default approximates it with tanh.
        activation_function="gelu", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        bos_token_id=tokenizer.eot_id, eos_token_id=tokenizer.eot_id, pad_token_id=tokenizer.pad_id,
    )
    ours = model.state_dict()
    state = {
        "transformer.wte.weight": ours["embed_tokens.weight"],
        "transformer.wpe.weight": ours["embed_positions.weight"],
        "transformer.ln_f.weight": ours["norm.weight"],
        "transformer.ln_f.bias": ours["norm.bias"],
    }
    for i in range(c.n_layer):
        b, h = f"blocks.{i}.", f"transformer.h.{i}."
        # GPT-2 projects q, k and v with one layer, three times as wide.
        state[h + "attn.c_attn.weight"] = torch.cat([ours[b + f"attn.{p}_proj.weight"] for p in "qkv"]).t()
        state[h + "attn.c_attn.bias"] = torch.cat([ours[b + f"attn.{p}_proj.bias"] for p in "qkv"])
        # Its layers store their weights as (in, out), the transpose of nn.Linear's (out, in).
        for theirs, mine in [("attn.c_proj", "attn.o_proj"), ("mlp.c_fc", "mlp.up_proj"), ("mlp.c_proj", "mlp.down_proj")]:
            state[h + theirs + ".weight"] = ours[b + mine + ".weight"].t()
            state[h + theirs + ".bias"] = ours[b + mine + ".bias"]
        for theirs, mine in [("ln_1", "attn_norm"), ("ln_2", "mlp_norm")]:
            state[h + theirs + ".weight"] = ours[b + mine + ".weight"]
            state[h + theirs + ".bias"] = ours[b + mine + ".bias"]
    gpt2 = GPT2LMHeadModel(config)
    # The head is tied to wte, as ours is to embed_tokens, so it comes with it.
    missing, unexpected = gpt2.load_state_dict(state, strict=False)
    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing, unexpected)
    return gpt2


def to_neox(model, tokenizer):
    # With rotary embeddings, the model is a GPT-NeoX: pre-LayerNorm blocks with biases, rotary embeddings
    # on all the channels of each head, in the "rotate half" form, and the exact GELU.
    c = model.config
    head_size = c.n_embd // c.n_head
    config = GPTNeoXConfig(
        vocab_size=c.vocab_size, hidden_size=c.n_embd, num_hidden_layers=c.n_layer, num_attention_heads=c.n_head,
        intermediate_size=4 * c.n_embd, hidden_act="gelu", rotary_pct=1.0, rotary_emb_base=10_000,
        max_position_embeddings=c.block_size, layer_norm_eps=1e-5, use_parallel_residual=False,
        attention_bias=True, attention_dropout=0.0, hidden_dropout=0.0, tie_word_embeddings=True,
        bos_token_id=tokenizer.eot_id, eos_token_id=tokenizer.eot_id, pad_token_id=tokenizer.pad_id,
    )
    ours = model.state_dict()
    state = {
        "gpt_neox.embed_in.weight": ours["embed_tokens.weight"],
        "gpt_neox.final_layer_norm.weight": ours["norm.weight"],
        "gpt_neox.final_layer_norm.bias": ours["norm.bias"],
    }
    for i in range(c.n_layer):
        b, h = f"blocks.{i}.", f"gpt_neox.layers.{i}."
        # GPT-NeoX projects q, k and v with one layer, each head's query, key and value side by side.
        for kind in ("weight", "bias"):
            qkv = torch.stack([ours[b + f"attn.{p}_proj.{kind}"].view(c.n_head, head_size, -1) for p in "qkv"], dim=1)
            state[h + f"attention.query_key_value.{kind}"] = qkv.reshape(3 * c.n_embd, -1).squeeze(-1)
        for theirs, mine in [("attention.dense", "attn.o_proj"), ("mlp.dense_h_to_4h", "mlp.up_proj"),
                             ("mlp.dense_4h_to_h", "mlp.down_proj"), ("input_layernorm", "attn_norm"),
                             ("post_attention_layernorm", "mlp_norm")]:
            state[h + theirs + ".weight"] = ours[b + mine + ".weight"]
            state[h + theirs + ".bias"] = ours[b + mine + ".bias"]
    neox = GPTNeoXForCausalLM(config)
    # The head is tied to embed_in, as ours is to embed_tokens, so it comes with it.
    missing, unexpected = neox.load_state_dict(state, strict=False)
    assert set(missing) <= {"embed_out.weight"} and not unexpected, (missing, unexpected)
    return neox


def to_hf_bpe(tokenizer):
    # Byte-level BPE writes each byte as a printable character, with GPT-2's table, so tokens are strings.
    chars = bytes_to_unicode()
    pieces = ["".join(chars[byte] for byte in token) for token in tokenizer.token_bytes]
    fast = Tokenizer(BPE({piece: i for i, piece in enumerate(pieces)}, [(pieces[a], pieces[b]) for a, b in tokenizer.merges]))
    # Our chunks, from bpe_reference.py's regex, on characters instead of bytes: anything past ASCII is a
    # letter, as the bytes of a multi-byte character are.
    chunk = r" ?[A-Za-z0-9\x{80}-\x{10FFFF}]+| ?[^A-Za-z0-9\x{80}-\x{10FFFF} \t\n\r\f\v]+|[ \t\n\r\f\v]+"
    fast.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(chunk), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    fast.decoder = decoders.ByteLevel()
    fast.add_special_tokens([PAD, EOT])
    return PreTrainedTokenizerFast(tokenizer_object=fast, pad_token=PAD, bos_token=EOT, eos_token=EOT)


def to_hf_tokenizer(tokenizer):
    if isinstance(tokenizer, BPETokenizer):
        return to_hf_bpe(tokenizer)
    vocab = {token: i for i, token in enumerate(tokenizer.id_to_token)}
    # WordPiece with no "##" pieces in its vocabulary: a word is in it, or it's [UNK], like ours.
    fast = Tokenizer(WordPiece(vocab, unk_token=UNK, max_input_chars_per_word=100))
    # Whitespace splits text on \w+|[^\w\s]+, the regex our tokenizer uses.
    fast.pre_tokenizer = pre_tokenizers.Whitespace()
    fast.decoder = decoders.WordPiece()
    fast.add_special_tokens(SPECIALS)
    return PreTrainedTokenizerFast(tokenizer_object=fast, unk_token=UNK, pad_token=PAD, bos_token=EOT, eos_token=EOT)


@torch.no_grad()
def check(model, tokenizer, folder):
    # Load the folder the way anyone else would, and compare it with our model and tokenizer.
    hf_model = AutoModelForCausalLM.from_pretrained(folder)
    hf_tokenizer = AutoTokenizer.from_pretrained(folder)
    with open(os.path.join(DATA_DIR, "instruct_valid.txt"), encoding="utf-8") as f:
        text = f.read(200_000)
    ids = tokenizer.encode(text)
    print(f"token ids match: {hf_tokenizer(text).input_ids == ids}, on {len(ids):,} tokens")
    x = torch.tensor([ids[: model.config.block_size]])
    difference = (model(x)[0] - hf_model(x).logits).abs().max().item()
    print(f"largest difference in the logits: {difference:.1e}")
    return ids


def export_onnx(folder):
    onnx_dir = os.path.join(folder, "onnx")
    # With past: the model takes the cached keys and values as inputs, and returns them with the new ones.
    main_export(folder, onnx_dir, task="text-generation-with-past")
    for name in os.listdir(onnx_dir):
        if not name.endswith(".onnx"):
            os.remove(os.path.join(onnx_dir, name))  # copies of the configs, which stay at the top
    # The weights as 8-bit integers, a quarter of the size, for a browser to download.
    quantize_dynamic(os.path.join(onnx_dir, "model.onnx"), os.path.join(onnx_dir, "model_quantized.onnx"),
                     weight_type=QuantType.QUInt8)


def onnx_loss(path, ids, config):
    session = onnxruntime.InferenceSession(path)
    x = np.array([ids[:-1]])
    feeds = {"input_ids": x, "attention_mask": np.ones_like(x), "position_ids": np.arange(x.shape[1])[None]}
    for layer in range(config.n_layer):
        for kind in ("key", "value"):
            # An empty cache, as there are no tokens before these.
            feeds[f"past_key_values.{layer}.{kind}"] = np.zeros((1, config.n_head, 0, config.n_embd // config.n_head), np.float32)
    logits = session.run(["logits"], feeds)[0][0]
    return F.cross_entropy(torch.from_numpy(logits), torch.tensor(ids[1:])).item()


def describe(model, tokenizer):
    # What the model card says about this model: the Hub shows the card of the latest push.
    c = model.config
    size = f"{c.n_layer} blocks of {c.n_head} heads, an embedding size of {c.n_embd}"
    if c.rope:
        architecture = (f"It has {model.num_params() / 1e6:.1f} million parameters, saved as a `GPTNeoXForCausalLM`: "
                        f"{size}, rotary position embeddings, GELU, and a head tied to the token embeddings.")
    else:
        architecture = (f"It has {model.num_params() / 1e6:.1f} million parameters and GPT-2's architecture, so it's "
                        f"saved as a `GPT2LMHeadModel`: {size}, {c.block_size} learned positions, GELU, and a head "
                        "tied to the token embeddings.")
    if isinstance(tokenizer, BPETokenizer):
        vocabulary = (f"Its vocabulary is {len(tokenizer):,} tokens: the 256 bytes, {len(tokenizer.merges):,} merges "
                      "learned by its own byte-level BPE tokenizer, written in C++, then padding and end of text, "
                      "saved here as a `tokenizer.json`.")
        limitations = "- Any text can be encoded, and decoding gives it back as it was."
    else:
        vocabulary = ("Its vocabulary is 10,000 words and runs of punctuation, from its own word-level tokenizer, "
                      "written here as a `tokenizer.json`.")
        limitations = ("- A word outside its vocabulary becomes `[UNK]`.\n"
                       "- The tokenizer drops whitespace, so `tokenizer.decode` puts spaces around some punctuation, "
                       "as in\n  `Sam ' s` and `Summary :`. `detokenize` in mvGPT's `tokenizer.py` joins them back.")
    return {"architecture": architecture, "vocabulary": vocabulary, "limitations": limitations,
            "context": c.block_size, "arch_tag": "gpt_neox" if c.rope else "gpt2"}


def push(folder, repo, tag, message, about, prompt):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_card.md"), encoding="utf-8") as f:
        card = Template(f.read()).substitute(repo=repo, tag=tag, message=message, prompt=json.dumps(prompt), **about)
    # The README of a model repo is its model card, the page the Hub shows for it.
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    commit = api.upload_folder(repo_id=repo, folder_path=folder, commit_message=message)
    # A tag names this commit, so revision=tag loads it, whatever is pushed after it.
    api.create_tag(repo, tag=tag, revision=commit.oid, tag_message=message)
    print(f"\npushed {commit.commit_url}\ntagged {tag}")


def main():
    parser = argparse.ArgumentParser(description="Save a checkpoint in the Hugging Face format, as a GPT-2, "
                                                 "or with rotary embeddings, a GPT-NeoX.")
    parser.add_argument("checkpoint")
    parser.add_argument("folder")
    parser.add_argument("--push", metavar="REPO", help="a model repo on the Hugging Face Hub, such as nenuadrian/mvGPT")
    parser.add_argument("--tag", help="the tag for this version, such as ep003-sft")
    parser.add_argument("--message", help="what this version is, for the commit and the model card")
    parser.add_argument("--prompt", default="Once upon a time", help="the prompt in the model card's example")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint)
    model.eval()
    (to_neox if model.config.rope else to_gpt2)(model, tokenizer).save_pretrained(args.folder)
    to_hf_tokenizer(tokenizer).save_pretrained(args.folder)
    print(f"saved {args.folder}: {', '.join(sorted(os.listdir(args.folder)))}")
    ids = check(model, tokenizer, args.folder)

    export_onnx(args.folder)
    window = ids[: model.config.block_size + 1]
    loss = model(torch.tensor([window[:-1]]), torch.tensor([window[1:]]))[1].item()
    print(f"\n{'PyTorch':<22}{'':>9}  loss {loss:.4f}")
    for name in ("model.onnx", "model_quantized.onnx"):
        path = os.path.join(args.folder, "onnx", name)
        print(f"{name:<22}{os.path.getsize(path) / 1e6:6.1f} MB  loss {onnx_loss(path, window, model.config):.4f}")
    if args.push:
        push(args.folder, args.push, args.tag, args.message, describe(model, tokenizer), args.prompt)


if __name__ == "__main__":
    main()
