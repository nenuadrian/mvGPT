import numpy as np
import pytest
import torch

pytest.importorskip("mojo.importer")  # requirements.txt installs Mojo on macOS: elsewhere, these tests are skipped

from checkpoint import save_checkpoint  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from mojo_model import from_torch, generate  # noqa: E402
from tokenizer import WordTokenizer  # noqa: E402


def tiny(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=4, n_embd=32, dropout=0.0)).eval()
    tokenizer = WordTokenizer(["[UNK]", "[PAD]", "<|endoftext|>"] + [f"w{i}" for i in range(37)])
    path = str(tmp_path / "tiny.pt")
    save_checkpoint(path, model, tokenizer)
    return model, path


def test_mojo_gives_the_logits_pytorch_gives(tmp_path):
    torch_model, path = tiny(tmp_path)
    mojo_model, _ = from_torch(path)
    ids = np.random.default_rng(0).integers(0, 40, 16)
    theirs = torch_model(torch.from_numpy(ids)[None])[0][0].detach().numpy()
    # A token at a time through the kernels, each against PyTorch's logits at its position.
    for pos, token in enumerate(ids):
        ours = mojo_model.step(int(token), pos)
        assert np.abs(ours - theirs[pos]).max() < 1e-4


def test_both_write_the_same_tokens(tmp_path):
    torch_model, path = tiny(tmp_path)
    mojo_model, _ = from_torch(path)
    ids = [5, 9, 12]
    mojo_ids = generate(mojo_model, ids, 20, greedy=True)
    torch_ids = torch_model.generate(torch.tensor([ids]), 20, top_k=1)[0].tolist()
    assert mojo_ids == torch_ids
