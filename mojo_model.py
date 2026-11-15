import argparse
import textwrap
import time

import numpy as np

import mojo.importer  # noqa: F401  (with this imported, `import kernels` compiles kernels.mojo into a module)

import kernels  # noqa: E402

# Our GPT writing on the GPU through Mojo kernels (kernels.mojo), from Python: a PyTorch checkpoint's weights go to
# the GPU once, and then each step runs one token through the model and gives back the next token's logits.


class MojoGPT:
    def __init__(self, model):
        config = model.config
        if (config.rope or config.rmsnorm or config.swiglu or config.n_kv_head or not config.causal
                or not config.bias):
            raise ValueError("kernels.mojo has episode one's blocks: learned positions, LayerNorm, GELU and biases")
        self.config = config
        self.gpt = kernels.GPT(flatten(model), config.vocab_size, config.block_size, config.n_layer, config.n_head,
                               config.n_embd)
        self.logits = np.zeros(config.vocab_size, dtype=np.float32)

    def step(self, token, pos):
        """The token at position pos through the model: the next token's logits, in an array the next step reuses."""
        self.gpt.step(token, pos, self.logits)
        return self.logits


def flatten(model):
    """The model's weights as one array of float32, in the order kernels.mojo reads them."""
    state = model.state_dict()
    # The head's weights are the token embeddings, and as it has no bias, zeros stand in for one.
    parts = [state["embed_tokens.weight"], np.zeros(model.config.vocab_size), state["embed_positions.weight"]]
    for i in range(model.config.n_layer):
        for name in ["attn_norm", "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp_norm",
                     "mlp.up_proj", "mlp.down_proj"]:
            parts += [state[f"blocks.{i}.{name}.weight"], state[f"blocks.{i}.{name}.bias"]]
    parts += [state["norm.weight"], state["norm.bias"]]
    return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])


def from_torch(path):
    """A PyTorch checkpoint of ours, on the GPU through Mojo, and the checkpoint's tokenizer."""
    from checkpoint import load_checkpoint

    model, tokenizer, _ = load_checkpoint(path)
    return MojoGPT(model), tokenizer


def generate(model, ids, max_new_tokens, temperature=1.0, top_k=None, eos_id=None, greedy=False, rng=None):
    rng = rng or np.random.default_rng()
    # The prompt goes in a token at a time, which fills the cache.
    for pos, token in enumerate(ids):
        logits = model.step(token, pos)
    out = list(ids)
    for _ in range(max_new_tokens):
        if len(out) >= model.config.block_size:
            break
        if greedy:
            token = int(np.argmax(logits))
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = np.where(logits < np.sort(logits)[-top_k], -np.inf, logits)
            probs = np.exp(logits - logits.max())
            token = int(rng.choice(len(probs), p=probs / probs.sum()))
        out.append(token)
        if token == eos_id:
            break
        logits = model.step(token, len(out) - 1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Write a story on the GPU, through our Mojo kernels.")
    parser.add_argument("checkpoint")
    parser.add_argument("prompt", nargs="?", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = from_torch(args.checkpoint)
    ids = tokenizer.encode(args.prompt)
    start = time.perf_counter()
    out = generate(model, ids, args.tokens, args.temperature, args.top_k, tokenizer.eot_id,
                   rng=np.random.default_rng(args.seed))
    seconds = time.perf_counter() - start
    print(textwrap.fill(tokenizer.decode(out), width=90))
    new = len(out) - len(ids)
    print(f"\n{new} new tokens in {seconds:.2f} s, {new / seconds:,.0f} tokens/s")


if __name__ == "__main__":
    main()
