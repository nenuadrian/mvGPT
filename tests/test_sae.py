import torch

from model import GPT, GPTConfig
from sae import stream


def tiny_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=4, n_head=4, n_embd=32, dropout=0.0, rope=True)).eval()


def test_the_stream_is_the_output_of_the_blocks_before_it():
    model = tiny_model()
    idx = torch.randint(50, (2, 12))
    outputs = []
    hook = model.blocks[1].register_forward_hook(lambda module, inputs, out: outputs.append(out))
    model(idx)
    hook.remove()
    assert torch.allclose(stream(model, idx, 2), outputs[0], atol=1e-6)
