"""Writes stories with a trained model."""
import torch


def write(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens, temperature, top_k, eos_id=tokenizer.eot_id)
    model.train(was_training)
    return tokenizer.decode(out[0].tolist())
