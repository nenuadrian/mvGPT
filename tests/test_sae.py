import torch

from model import GPT, GPTConfig
from sae import SparseAutoencoder, Splice, stream


def tiny_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=4, n_head=4, n_embd=32, dropout=0.0, rope=True)).eval()


def test_each_token_keeps_k_features_at_most_and_none_negative():
    sae = SparseAutoencoder(32, 256, k=8)
    features = sae.encode(torch.randn(10, 32))
    assert ((features > 0).sum(dim=-1) <= 8).all()
    assert (features >= 0).all()


def test_a_feature_fires_on_its_own_direction_first():
    torch.manual_seed(0)
    sae = SparseAutoencoder(32, 256, k=8)
    # The decoder starts with each direction at a length of 1, and the encoder as its transpose.
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(256), atol=1e-5)
    features = sae.encode(sae.bias + 3 * sae.decoder.weight[:, 17])
    assert features.argmax().item() == 17


def test_the_stream_is_the_output_of_the_blocks_before_it():
    model = tiny_model()
    idx = torch.randint(50, (2, 12))
    outputs = []
    hook = model.blocks[1].register_forward_hook(lambda module, inputs, out: outputs.append(out))
    model(idx)
    hook.remove()
    assert torch.allclose(stream(model, idx, 2), outputs[0], atol=1e-6)


def test_a_splice_changes_what_the_rest_of_the_model_sees_and_goes_away():
    model = tiny_model()
    idx = torch.randint(50, (1, 8))
    clean = model(idx)[0]
    with Splice(model, 2, lambda s: s):
        assert torch.allclose(model(idx)[0], clean)
    with Splice(model, 2, lambda s: torch.zeros_like(s)):
        assert not torch.allclose(model(idx)[0], clean)
    assert torch.allclose(model(idx)[0], clean)
