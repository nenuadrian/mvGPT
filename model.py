"""mvGPT: a minimum viable GPT, a decoder-only transformer you can read end to end."""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 10_000
    block_size: int = 256  # the context window, in tokens
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384  # split between the heads: 6 x 64
    dropout: float = 0.1
    bias: bool = True
