"""Supervised fine-tuning: the pretrained model learns to write the story an instruction asks for.

    python sft.py 10
"""
import re

from tokenizer import EOT

# The fields of an instruction, in the order our prompts give them, before the story.
FIELDS = ["Features", "Words", "Random sentence", "Summary"]
FIELD = re.compile(r"^(Features|Words|Random sentence|Summary|Story):", re.MULTILINE)


def parse(doc):
    # The split alternates names and texts: ["", "Words", " dog, park\n", "Story", " Once ...", ...].
    parts = FIELD.split(doc)
    return {name: text.strip() for name, text in zip(parts[1::2], parts[2::2])}


def instruction(fields):
    lines = [f"{name}: {fields[name]}" for name in FIELDS if name in fields]
    return "\n".join(lines + ["Story:"])


def build_examples(path, tokenizer, block_size, limit=None):
    with open(path, encoding="utf-8") as f:
        docs = [d for d in f.read().split(EOT) if d.strip()][:limit]
    examples = []
    for doc in docs:
        fields = parse(doc)
        if not fields.get("Story"):
            continue
        prompt = tokenizer.encode(instruction(fields))
        story = tokenizer.encode(fields["Story"]) + [tokenizer.eot_id]
        ids = (prompt + story)[: block_size + 1]
        # An instruction that fills the window leaves nothing to learn from.
        if len(ids) > len(prompt):
            examples.append((ids, len(prompt)))
    return examples
