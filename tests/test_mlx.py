import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")  # requirements.txt installs MLX on macOS: elsewhere, these tests are skipped

from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from mlx_model import from_torch, generate, to_torch  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import WordTokenizer  # noqa: E402


def tiny(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=4, n_embd=32, dropout=0.0)).eval()
    tokenizer = WordTokenizer(["[UNK]", "[PAD]", "<|endoftext|>"] + [f"w{i}" for i in range(37)])
    path = str(tmp_path / "tiny.pt")
    save_checkpoint(path, model, tokenizer)
    return model, path


def test_mlx_gives_the_logits_pytorch_gives(tmp_path):
    torch_model, path = tiny(tmp_path)
    mlx_model, _ = from_torch(path)
    idx = np.random.default_rng(0).integers(0, 40, (2, 16))
    ours = np.array(mlx_model(mx.array(idx)))
    theirs = torch_model(torch.from_numpy(idx))[0].detach().numpy()
    assert np.abs(ours - theirs).max() < 1e-4


def test_both_write_the_same_tokens_and_the_cache_changes_nothing(tmp_path):
    torch_model, path = tiny(tmp_path)
    mlx_model, _ = from_torch(path)
    ids = [5, 9, 12]
    mlx_ids = generate(mlx_model, ids, 20, greedy=True)
    torch_ids = torch_model.generate(torch.tensor([ids]), 20, top_k=1)[0].tolist()
    assert mlx_ids == torch_ids


def test_a_model_goes_to_pytorch_and_back_unchanged(tmp_path):
    _, path = tiny(tmp_path)
    mlx_model, tokenizer = from_torch(path)
    back = str(tmp_path / "back.pt")
    to_torch(mlx_model, tokenizer, back)
    before, _, _ = load_checkpoint(path)
    after, _, _ = load_checkpoint(back)
    for name, weight in before.state_dict().items():
        assert torch.equal(weight, after.state_dict()[name])
