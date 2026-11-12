import math

import mlx.core as mx
import mlx.nn as nn

# Our GPT in MLX, Apple's array library for its own chips: episode one's blocks, with LayerNorm, GELU and a learned
# embedding per position, under the names model.py gives them, so that a PyTorch checkpoint's weights load by name.


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd, bias = config["n_embd"], config["bias"]
        self.n_head = config["n_head"]
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(config["dropout"])

    def __call__(self, x, cache=None):
        B, T, C = x.shape
        # Batch, heads, tokens, channels per head, as in model.py.
        q = self.q_proj(x).reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            # The cache is this layer's keys and values so far, a list we extend in place.
            if cache:
                k = mx.concatenate([cache[0], k], axis=2)
                v = mx.concatenate([cache[1], v], axis=2)
            cache[:] = [k, v]
        # A single new token sees all the cached ones, so it needs no mask.
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(q.shape[-1]),
                                                 mask="causal" if T > 1 else None)
        return self.dropout(self.o_proj(y.transpose(0, 2, 1, 3).reshape(B, T, C)))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config["n_embd"], 4 * config["n_embd"], bias=config["bias"])
        self.down_proj = nn.Linear(4 * config["n_embd"], config["n_embd"], bias=config["bias"])
        self.dropout = nn.Dropout(config["dropout"])

    def __call__(self, x):
        # MLX's gelu is the exact one, with the error function, like ours.
        return self.dropout(self.down_proj(nn.gelu(self.up_proj(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config["n_embd"], bias=config["bias"])
        self.attn = Attention(config)
        self.mlp_norm = nn.LayerNorm(config["n_embd"], bias=config["bias"])
        self.mlp = MLP(config)

    def __call__(self, x, cache=None):
        x = x + self.attn(self.attn_norm(x), cache)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (config.get("rope") or config.get("rmsnorm") or config.get("swiglu") or config.get("n_kv_head")
                or not config.get("causal", True)):
            raise ValueError("mlx_model.py has episode one's blocks: learned positions, LayerNorm, GELU, causal")
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.embed_positions = nn.Embedding(config["block_size"], config["n_embd"])
        self.drop = nn.Dropout(config["dropout"])
        self.blocks = [Block(config) for _ in range(config["n_layer"])]
        self.norm = nn.LayerNorm(config["n_embd"], bias=config["bias"])

    def __call__(self, idx, cache=None):
        B, T = idx.shape
        # With a cache, these tokens come after the ones in it.
        start = cache[0][0].shape[2] if cache and cache[0] else 0
        x = self.drop(self.embed_tokens(idx) + self.embed_positions(mx.arange(start, start + T)))
        for i, block in enumerate(self.blocks):
            x = block(x, None if cache is None else cache[i])
        # The head shares its weights with the token embeddings, as in model.py: as_linear multiplies by them.
        return self.embed_tokens.as_linear(self.norm(x))


def from_torch(path):
    """A PyTorch checkpoint of ours, as an MLX model and the checkpoint's tokenizer."""
    import torch

    from checkpoint import load_checkpoint

    torch_model, tokenizer, ckpt = load_checkpoint(path)
    config = dict(ckpt["config"])
    model = GPT(config)
    # Each PyTorch tensor through NumPy into an MLX array, under its own name. The head is the embeddings, so its copy
    # is left out.
    weights = [(name, mx.array(t.numpy())) for name, t in torch_model.state_dict().items() if name != "lm_head.weight"]
    model.load_weights(weights)
    model.eval()
    return model, tokenizer


def generate(model, ids, max_new_tokens, temperature=1.0, top_k=None, eos_id=None, greedy=False):
    cache = [[] for _ in model.blocks]
    logits = model(mx.array([ids]), cache)
    out = list(ids)
    for _ in range(max_new_tokens):
        if len(out) >= model.config["block_size"]:
            break
        logits = logits[:, -1, :]
        if greedy:
            next_id = mx.argmax(logits, axis=-1)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = mx.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = mx.where(logits < kth, -mx.inf, logits)
            next_id = mx.random.categorical(logits)
        # item() waits for the GPU: MLX is lazy, and builds the work up until a value is needed.
        token = next_id.item()
        out.append(token)
        if token == eos_id:
            break
        logits = model(next_id[:, None], cache)
    return out


def to_torch(model, tokenizer, path):
    """An MLX model of ours back as a PyTorch checkpoint, so that sample.py and evaluate.py load it."""
    import numpy as np
    import torch
    from mlx.utils import tree_flatten

    from checkpoint import save_checkpoint
    from model import GPT as TorchGPT, GPTConfig

    state = {name: torch.from_numpy(np.array(w)) for name, w in tree_flatten(model.parameters())}
    state["lm_head.weight"] = state["embed_tokens.weight"]
    torch_model = TorchGPT(GPTConfig(**model.config))
    torch_model.load_state_dict(state)
    save_checkpoint(path, torch_model, tokenizer)
