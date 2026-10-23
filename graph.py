import json
import os
import urllib.error
import urllib.parse
import urllib.request

import torch

from model import KVCache
from tools import show

# A call is a query in smollerGraph's Cypher, as plain text: the model writes
# <<cypher MATCH (s:Species {name: 'Red fox'})-[:PARENT_TAXON*2]->(f:Family) RETURN f.name>>, and the program running
# it, which watches for the >>, sends the query to sg serve and appends " = Canidae". sg serve answers from the graph
# on disk, and nothing it serves changes the graph, so a query the model writes can only read.
OPEN, CLOSE = "<<cypher ", ">>"
SERVER = os.environ.get("SMOLLERGRAPH", "http://localhost:8474")
SHOWN = 5  # rows that go back into the context; the rest are counted, so a long answer can't fill the window
LONGEST = 300  # characters in a query


def query(cypher, server=SERVER):
    """sg serve's answer, as JSON: the columns, the rows and how many there are, or what's wrong with the query."""
    url = f"{server}/api/query?" + urllib.parse.urlencode({"q": cypher})
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # A query sg can't parse or plan is a 400, with the reason in it.
        return json.load(error)


def cell(value):
    # A property the node doesn't have comes back as null.
    if value is None:
        return "unknown"
    return show(value) if isinstance(value, float) else str(value)


def result_text(answer):
    """What goes in after the call: the first rows, a row's values joined by |, and how many more there are."""
    if "error" in answer:
        return f"error: {answer['error']}"
    rows = [" | ".join(cell(c["value"]) for c in row) for row in answer["rows"][:SHOWN]]
    if not rows:
        return "nothing"
    more = answer["count"] - len(rows)
    return ", ".join(rows) + (f" and {more} more" if more > 0 else "")


def run_query(cypher, server=SERVER):
    if len(cypher) > LONGEST:
        return "error: too long"
    return result_text(query(cypher, server))


def literal(text):
    # A string in Cypher, in single quotes, or in double ones when it has an apostrophe, as Finsch's Francolin does.
    return f'"{text}"' if "'" in text else f"'{text}'"


@torch.no_grad()
def generate_with_graph(model, tokenizer, ids, max_new_tokens=192, use_tool=True, server=SERVER):
    """Writes after ids, one token at a time; when a call closes, sg serve's answer goes in after it.
    Returns the new ids and the calls, each (query, result). use_tool=False lets the model write its own."""
    model.eval()
    device = next(model.parameters()).device
    cache = KVCache(model.config.n_layer)
    logits, _ = model(torch.tensor([ids], device=device), cache=cache)
    out, calls, opened = [], [], None
    while len(out) < max_new_tokens and len(ids) + len(out) < model.config.block_size:
        token = logits[0, -1].argmax().item()
        if token == tokenizer.eot_id:
            break
        out.append(token)
        new = [token]
        text = tokenizer.decode(out)
        if opened is None and text.endswith(OPEN.strip()):
            opened = len(text)
        # Cypher has > in its arrows and comparisons, but never two together.
        if opened is not None and text.endswith(CLOSE):
            cypher = text[opened:-len(CLOSE)].strip()
            opened = None
            if use_tool:
                result = run_query(cypher, server)
                calls.append((cypher, result))
                tool_ids = tokenizer.encode(f" = {result}")
                out.extend(tool_ids)
                new += tool_ids
            else:
                calls.append((cypher, None))
        # The newest tokens alone, the tool's included: the cache has the keys and values of the ones before.
        logits, _ = model(torch.tensor([new], device=device), cache=cache)
    return out, calls


def main():
    import argparse

    from checkpoint import load_checkpoint
    from model import pick_device

    parser = argparse.ArgumentParser(description="Ask a model a question, with smollerGraph to look facts up in.")
    parser.add_argument("checkpoint")
    parser.add_argument("question")
    parser.add_argument("--no-tool", action="store_true", help="let the model write the results of its queries itself")
    parser.add_argument("--server", default=SERVER, help="where sg serve is listening")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    prompt = f"Question: {args.question}\nAnswer:"
    ids = tokenizer.encode(prompt)
    out, calls = generate_with_graph(model, tokenizer, ids, use_tool=not args.no_tool, server=args.server)
    print(prompt + tokenizer.decode(out))
    for cypher, result in calls:
        print(f"  smollerGraph: {cypher} -> {result if result is not None else '(off)'}")


if __name__ == "__main__":
    main()
