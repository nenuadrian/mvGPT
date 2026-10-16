import argparse
import json
import os
import urllib.request

import torch
from torch.nn import functional as F

from checkpoint import load_checkpoint
from data import DATA_DIR
from model import pick_device

# HellaSwag's validation set, from its copy on Hugging Face: 10,042 beginnings of a scene, from ActivityNet
# captions and WikiHow, each with four endings to choose from. People pick the right one about 95% of the time;
# guessing gets 25%.
URL = "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/validation-00000-of-00001.parquet"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_examples(limit=None):
    import pyarrow.parquet as pq

    path = os.path.join(DATA_DIR, "hellaswag_val.parquet")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    examples = pq.read_table(path, columns=["ctx", "endings", "label"]).to_pylist()
    for example in examples:
        example["label"] = int(example["label"])
    return examples[:limit]


@torch.no_grad()
def ending_losses(forward, encode, example, device):
    # The four endings after the same beginning, a row each, padded to the longest. Only the ending's tokens
    # are scored: how surprised the model is by each ending, given the beginning.
    context = encode(example["ctx"])
    endings = [encode(" " + ending) for ending in example["endings"]]
    rows = [context + ending for ending in endings]
    width = max(map(len, rows))
    x = torch.zeros(4, width, dtype=torch.long)
    scored = torch.zeros(4, width, dtype=torch.bool)
    for i, (row, ending) in enumerate(zip(rows, endings)):
        x[i, : len(row)] = torch.tensor(row)
        scored[i, len(context) : len(row)] = True
    x, scored = x.to(device), scored.to(device)
    logits = forward(x)
    # Each position predicts the next token, so the losses line up with tokens 1 to the end.
    losses = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), x[:, 1:].flatten(), reduction="none").view(4, -1)
    losses = losses * scored[:, 1:]
    # The sum favours short endings, so the score that counts is the average over each ending's tokens.
    return losses.sum(1), losses.sum(1) / scored[:, 1:].sum(1)


def main():
    parser = argparse.ArgumentParser(description="How often a model picks HellaSwag's right ending.")
    parser.add_argument("checkpoint", help="one of ours, or gpt2 for OpenAI's GPT-2 small, from Hugging Face")
    parser.add_argument("--limit", type=int, help="only the first examples")
    args = parser.parse_args()

    device = pick_device()
    if args.checkpoint == "gpt2":
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        forward, encode, name = (lambda x: model(x).logits), tokenizer.encode, "gpt2"
        params = model.num_parameters()
    else:
        model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
        model.eval()
        forward, encode = (lambda x: model(x)[0]), tokenizer.encode
        name = os.path.basename(os.path.dirname(args.checkpoint))
        params = model.num_params()

    examples = load_examples(args.limit)
    correct = correct_norm = 0
    for i, example in enumerate(examples):
        total, average = ending_losses(forward, encode, example, device)
        correct += total.argmin().item() == example["label"]
        correct_norm += average.argmin().item() == example["label"]
        if (i + 1) % 1000 == 0:
            print(f"{i + 1:>6,} examples | accuracy {correct_norm / (i + 1):.2%}", flush=True)
    results = {"model": name, "parameters": params, "examples": len(examples),
               "accuracy": correct_norm / len(examples), "accuracy_sum": correct / len(examples)}
    print(f"{name}: {results['accuracy']:.2%} of {len(examples):,} right, by the average loss of each ending "
          f"({results['accuracy_sum']:.2%} by the total)")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"hellaswag-{name}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
