import torch
from transformers import GPT2Config, GPT2LMHeadModel

from server import Request, write

PAD, EOS = 0, 1


def requests(prompts):
    # A top k of 1 always picks the likeliest token, so what's written doesn't depend on chance.
    return [Request(ids, max_tokens=12, temperature=1.0, top_k=1) for ids in prompts]


def test_batch_writes_what_each_request_writes_alone():
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=50, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    # Prompts of different lengths, so the batch pads some of them.
    prompts = [torch.randint(2, 50, (n,)).tolist() for n in (3, 9, 5)]
    alone = []
    for r in requests(prompts):
        write(model, [r], PAD, EOS)
        alone.append(r.tokens)
    batch = requests(prompts)
    write(model, batch, PAD, EOS)
    assert [r.tokens for r in batch] == alone
