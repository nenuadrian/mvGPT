import ast
import operator
import re

import torch

from model import KVCache

# A call is plain text: the model writes <<calc 37 * 24>>, and the program running it, which watches for the >>,
# appends " = 888" and lets the model carry on. The vocabulary doesn't change.
OPEN, CLOSE = "<<calc ", ">>"
CALL = re.compile(r"<<calc ([^<>]*)>>")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression, longest=100):
    # Numbers, + - * /, brackets and minus signs, and nothing else: a call the model writes can't run other code.
    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        raise ValueError(f"not arithmetic: {ast.dump(node)}")

    if len(expression) > longest:
        return "error"
    try:
        return show(value(ast.parse(expression.strip(), mode="eval").body))
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return "error"


def show(number):
    # Whole numbers without a decimal point, others to 4 places, the way the examples write them.
    if isinstance(number, float):
        number = round(number, 4)
        if number == int(number):
            number = int(number)
    return str(number)


def result_text(expression):
    return f" = {calculate(expression)}"


@torch.no_grad()
def generate_with_tools(model, tokenizer, ids, max_new_tokens=64, use_tool=True, temperature=0.0):
    """Writes after ids, one token at a time; when a call closes, the calculator's result goes in after it.
    Returns the new ids and the calls, each (expression, result). use_tool=False lets the model write its own."""
    model.eval()
    device = next(model.parameters()).device
    cache = KVCache(model.config.n_layer)
    logits, _ = model(torch.tensor([ids], device=device), cache=cache)
    out, calls, opened = [], [], None
    while len(out) < max_new_tokens and len(ids) + len(out) < model.config.block_size:
        last = logits[0, -1]
        if temperature > 0:
            token = torch.multinomial(torch.softmax(last / temperature, dim=-1), 1).item()
        else:
            token = last.argmax().item()
        if token == tokenizer.eot_id:
            break
        out.append(token)
        new = [token]
        text = tokenizer.decode(out)
        if opened is None and text.endswith(OPEN.strip()):
            opened = len(text)
        if opened is not None and text.endswith(CLOSE):
            # The call is closed: what's between the markers goes to the calculator.
            expression = text[opened:-len(CLOSE)].strip()
            opened = None
            if use_tool:
                result = result_text(expression)
                calls.append((expression, result[3:]))
                tool_ids = tokenizer.encode(result)
                out.extend(tool_ids)
                new += tool_ids
            else:
                calls.append((expression, None))
        # The newest tokens alone, the tool's included: the cache has the keys and values of the ones before.
        logits, _ = model(torch.tensor([new], device=device), cache=cache)
    return out, calls


def main():
    import argparse

    from checkpoint import load_checkpoint
    from model import pick_device

    parser = argparse.ArgumentParser(description="Ask a model a question, with the calculator.")
    parser.add_argument("checkpoint")
    parser.add_argument("question")
    parser.add_argument("--no-tool", action="store_true", help="let the model write the results of its calls itself")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    prompt = f"Question: {args.question}\nAnswer:"
    out, calls = generate_with_tools(model, tokenizer, tokenizer.encode(prompt), use_tool=not args.no_tool)
    print(prompt + tokenizer.decode(out))
    for expression, result in calls:
        print(f"  calculator: {expression} -> {result if result is not None else '(off)'}")


if __name__ == "__main__":
    main()
