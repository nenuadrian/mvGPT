"""A trained model in one file: its weights, its config and its vocabulary."""
import os
from dataclasses import asdict

import torch

from model import GPT, GPTConfig
from tokenizer import WordTokenizer

CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def save_checkpoint(path, model, tokenizer, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model": model.state_dict(), "config": asdict(model.config), "vocab": tokenizer.id_to_token}
    torch.save({**state, **meta}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, WordTokenizer(ckpt["vocab"]), ckpt
