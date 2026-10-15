import math

import torch

from model import GPT, GPTConfig

torch.manual_seed(1337)
config = GPTConfig()
model = GPT(config)
print(model)

# lm_head isn't listed: it shares embed_tokens' weight, which PyTorch counts once.
counts = {}
for name, p in model.named_parameters():
    layer = name.split(".")[-2]
    counts[layer] = counts.get(layer, 0) + p.numel()
print()
for layer, n in counts.items():
    print(f"{layer:<16}{n:>12,}")
print(f"{'total':<16}{model.num_params():>12,}")

model.eval()
x = torch.randint(config.vocab_size, (2, 4))
y = torch.randint(config.vocab_size, (2, 4))
watch = ["embed_tokens", "embed_positions", *(f"blocks.{i}" for i in range(config.n_layer)), "norm", "lm_head"]
hooks = [
    model.get_submodule(name).register_forward_hook(
        lambda module, inputs, output, name=name: print(f"{name:<16}{tuple(output.shape)}")
    )
    for name in watch
]
print()
print(f"{'tokens':<16}{tuple(x.shape)}")
logits, loss = model(x, y)
for hook in hooks:
    hook.remove()
# Before training, each of the 10,000 tokens is about as likely as any other.
print(f"loss {loss.item():.2f}, ln({config.vocab_size:,}) = {math.log(config.vocab_size):.2f}")

# "Once upon a time" as token ids: there's no tokenizer yet.
prompt = torch.tensor([[60, 61, 7, 47]])
out = model.generate(prompt, max_new_tokens=12)
print()
print(out[0].tolist())
