import torch

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from tools import calculate, generate_with_tools


def test_arithmetic():
    assert calculate("37 * 24") == "888"
    assert calculate("(3 + 4) * -2") == "-14"
    assert calculate("120 / 8") == "15"
    assert calculate("10 / 4") == "2.5"
    assert calculate("1 / 3") == "0.3333"
    assert calculate(" 999999 * 999999 ") == "999998000001"


def test_refuses_what_isnt_arithmetic():
    for expression in ["__import__('os').system('ls')", "2 ** 1000", "abs(-3)", "x + 1", "1 / 0", "3 +", "", "1" * 101]:
        assert calculate(expression) == "error", expression


class Scripted:
    """Stands in for a model: writes the tokens of a script, one per step, whatever it's given."""

    def __init__(self, tokenizer, script):
        self.config = GPTConfig(vocab_size=len(tokenizer), n_layer=1, block_size=256)
        self.script, self.fed = list(script), []
        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.param])

    def eval(self):
        return self

    def __call__(self, idx, cache=None):
        self.fed.append(idx[0].tolist())
        logits = torch.full((1, idx.size(1), self.config.vocab_size), -1e9)
        logits[0, -1, self.script.pop(0) if self.script else 0] = 0
        return logits, None


def test_the_result_goes_in_after_the_call():
    tokenizer = BPETokenizer([])  # bytes alone: each character is a token
    script = tokenizer.encode(" <<calc 37 * 24>>") + tokenizer.encode(". Done.") + [tokenizer.eot_id]
    model = Scripted(tokenizer, script)
    out, calls = generate_with_tools(model, tokenizer, tokenizer.encode("Answer:"))
    assert calls == [("37 * 24", "888")]
    assert tokenizer.decode(out).startswith(" <<calc 37 * 24>> = 888")
    # The model reads the tool's tokens along with the >> it wrote, in one step.
    assert [tokenizer.decode(f) for f in model.fed if len(f) > 1][-1] == "> = 888"


def test_without_the_tool_the_model_writes_its_own():
    tokenizer = BPETokenizer([])
    script = tokenizer.encode(" <<calc 2 + 2>> = 5") + [tokenizer.eot_id]
    out, calls = generate_with_tools(Scripted(tokenizer, script), tokenizer, tokenizer.encode("A:"), use_tool=False)
    assert calls == [("2 + 2", None)]
    assert tokenizer.decode(out) == " <<calc 2 + 2>> = 5"


def test_a_real_model_runs():
    tokenizer = BPETokenizer([])
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(tokenizer), n_layer=2, n_head=2, n_embd=32, block_size=64, rope=True))
    out, _ = generate_with_tools(model, tokenizer, tokenizer.encode("Q: 1 + 1\nA:"), max_new_tokens=10)
    assert len(out) <= 10
