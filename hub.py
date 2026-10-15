import argparse
import os

import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.exporters.onnx import main_export
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import WordPiece
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from checkpoint import load_checkpoint
from data import DATA_DIR
from tokenizer import EOT, PAD, SPECIALS, UNK


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


def to_hf_tokenizer(tokenizer):
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


def main():
    parser = argparse.ArgumentParser(description="Save a checkpoint in the Hugging Face format, as a GPT-2.")
    parser.add_argument("checkpoint")
    parser.add_argument("folder")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint)
    model.eval()
    to_gpt2(model, tokenizer).save_pretrained(args.folder)
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


if __name__ == "__main__":
    main()
