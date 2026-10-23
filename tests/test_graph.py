import pytest

import graph
from graph import generate_with_graph, literal, result_text
from test_tools import Scripted
from tokenizer import BPETokenizer


def answer(*rows, count=None):
    # sg serve's JSON for some rows of values, each value with the id of its node.
    rows = [[{"id": 0, "value": value} for value in row] for row in rows]
    return {"count": len(rows) if count is None else count, "columns": [], "rows": rows}


def test_results():
    assert result_text(answer(["Canidae"])) == "Canidae"
    assert result_text(answer(["least concern", 157000])) == "least concern | 157000"
    assert result_text(answer(["vulnerable", None])) == "vulnerable | unknown"
    assert result_text(answer(["Tiger"], ["Flat-headed cat"])) == "Tiger, Flat-headed cat"
    assert result_text(answer()) == "nothing"
    assert result_text({"error": "expected a variable at 34, not the end"}) == "error: expected a variable at 34, not the end"


def test_long_answers_are_counted():
    names = [[f"species {i}"] for i in range(5)]
    assert result_text(answer(*names, count=17)) == "species 0, species 1, species 2, species 3, species 4 and 12 more"


def test_quotes():
    assert literal("Red fox") == "'Red fox'"
    assert literal("Finsch's Francolin") == '"Finsch\'s Francolin"'


def test_the_answer_goes_in_after_the_query(monkeypatch):
    asked = []
    monkeypatch.setattr(graph, "query", lambda cypher, server: asked.append(cypher) or answer(["Canidae"]))
    tokenizer = BPETokenizer([])  # bytes alone: each character is a token
    cypher = "MATCH (s {name: 'Red fox'})-[:PARENT_TAXON*2]->(f:Family) WHERE s.weight_g > 10 RETURN f.name"
    script = tokenizer.encode(f" <<cypher {cypher}>>") + tokenizer.encode(". Canidae.") + [tokenizer.eot_id]
    model = Scripted(tokenizer, script)
    model.config.block_size = 1024
    out, calls = generate_with_graph(model, tokenizer, tokenizer.encode("Answer:"), max_new_tokens=200)
    # The query's arrow and its > don't close the call: only >> does.
    assert asked == [cypher]
    assert calls == [(cypher, "Canidae")]
    assert tokenizer.decode(out) == f" <<cypher {cypher}>> = Canidae. Canidae."


def test_without_the_tool_the_model_writes_its_own(monkeypatch):
    monkeypatch.setattr(graph, "query", lambda cypher, server: pytest.fail("the graph was asked"))
    tokenizer = BPETokenizer([])
    script = tokenizer.encode(" <<cypher MATCH (s) RETURN s.name>> = Lion") + [tokenizer.eot_id]
    out, calls = generate_with_graph(Scripted(tokenizer, script), tokenizer, tokenizer.encode("A:"), use_tool=False)
    assert calls == [("MATCH (s) RETURN s.name", None)]
    assert tokenizer.decode(out) == " <<cypher MATCH (s) RETURN s.name>> = Lion"


def test_a_long_query_is_refused(monkeypatch):
    monkeypatch.setattr(graph, "query", lambda cypher, server: pytest.fail("the graph was asked"))
    assert graph.run_query("MATCH " + "(a)-[:R]->" * 40 + "(b) RETURN b.name") == "error: too long"
