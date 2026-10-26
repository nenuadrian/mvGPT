import random
import re

import agent
import graph_data
from agent_data import show
from graph_agent_data import Maker, as_role, handed_back, with_episode21_tools
from graph_data import Maker as GraphMaker
from graph_data import read_facts
from test_agent import agent as scripted_agent
from tools import calculate

# A small graph, in smollerGraph's facts: cats and a bear, two of them without a weight. None of these names is
# held out, so the training side of the makers sees them all.
FACTS = """(Mammalia:Class {taxon: 'Mammalia'})
(Carnivora:Order {taxon: 'Carnivora'})
(Felidae:Family {taxon: 'Felidae'})
(Ursidae:Family {taxon: 'Ursidae'})
(Panthera:Genus {taxon: 'Panthera'})
(Acinonyx:Genus {taxon: 'Acinonyx'})
(Ursus:Genus {taxon: 'Ursus'})
(Lion:Species {taxon: 'Panthera leo', status: 'vulnerable', weight_g: 157000})
(Tiger:Species {taxon: 'Panthera tigris', status: 'endangered', weight_g: 150000})
(Cheetah:Species {taxon: 'Acinonyx jubatus', status: 'vulnerable', weight_g: 37600})
(Brown bear:Species {taxon: 'Ursus arctos', status: 'least concern', weight_g: 217000})
(Snow leopard:Species {taxon: 'Panthera uncia', status: 'vulnerable'})
(Jaguar:Species {taxon: 'Panthera onca', status: 'near threatened'})
(Lion)-[:PARENT_TAXON]->(Panthera)
(Tiger)-[:PARENT_TAXON]->(Panthera)
(Snow leopard)-[:PARENT_TAXON]->(Panthera)
(Jaguar)-[:PARENT_TAXON]->(Panthera)
(Cheetah)-[:PARENT_TAXON]->(Acinonyx)
(Brown bear)-[:PARENT_TAXON]->(Ursus)
(Panthera)-[:PARENT_TAXON]->(Felidae)
(Acinonyx)-[:PARENT_TAXON]->(Felidae)
(Ursus)-[:PARENT_TAXON]->(Ursidae)
(Felidae)-[:PARENT_TAXON]->(Carnivora)
(Ursidae)-[:PARENT_TAXON]->(Carnivora)
(Carnivora)-[:PARENT_TAXON]->(Mammalia)
"""


def sg(nodes, parent):
    """Stands in for sg serve, as graph.run_query's text: the queries graph_data.py writes about one species."""

    def run_query(cypher, server=None):
        name = re.search(r"name: (?:'([^']*)'|\"([^\"]*)\")", cypher)
        name = name[1] or name[2]
        if name not in nodes:
            return "nothing"
        if hops := re.search(r"PARENT_TAXON\*(\d)", cypher):
            node = name
            for _ in range(int(hops[1])):
                node = parent[node]
            return node
        key = cypher.rsplit("s.", 1)[1]
        value = nodes[name][1].get(key)
        return "unknown" if value is None else str(value)

    return run_query


def makers(tmp_path, monkeypatch, seed=0):
    path = tmp_path / "facts.txt"
    path.write_text(FACTS)
    nodes, parent = read_facts(path)
    fake = sg(nodes, parent)
    monkeypatch.setattr(graph_data, "run_query", fake)  # what the data builder asks
    monkeypatch.setattr(agent, "run_query", fake)  # what the agent's graph role asks
    rng = random.Random(seed)
    g = GraphMaker(nodes, parent, rng)
    return Maker(g, parent, rng), g


def replay(tmp_path, example, math=()):
    """Runs a generated example through the agent, with a model that writes the example's own learned text for the
    agent, the graph subagents' examples for the graph role, and math's scripts: the runtime adds every result."""

    def learned(e):
        return "".join(text for text, learn in e["segments"] if learn).replace("<|endoftext|>", "")

    scripts = {"agent": [learned(example)], "graph": [learned(s) for s in example["subagents"]], "math": list(math)}
    a, box = scripted_agent(tmp_path, scripts)
    return a.run("agent", example["task"]), box


def results(example):
    # The results in an example's trace, in order: what follows each call.
    s = example["segments"]
    return [s[i + 1][0].removeprefix(" = ") for i, (text, learn) in enumerate(s[:-1]) if learn and "<<" in text]


def test_heavier_asks_the_graph_twice_then_math(tmp_path, monkeypatch):
    maker, _ = makers(tmp_path, monkeypatch)
    for _ in range(20):
        e = maker.make("heavier", "known")
        calls = [text for text, learn in e["segments"] if learn and "<<" in text]
        a, b = (float(r.split("about ")[1].split(" ")[0]) for r in results(e)[:2])
        assert a > b  # the heavier one is named first, so the difference is positive
        assert [c.split(":")[0].strip(" .<") for c in calls] == ["ask graph", "ask graph", "ask math"]
        assert results(e)[2] == e["expect"]["answer"]
    lion_tiger = [e for e in (maker.make("heavier", "known") for _ in range(200)) if "the Lion than the Tiger" in e["task"]]
    e = lion_tiger[0]
    assert results(e) == ["The Lion weighs about 157000 grams", "The Tiger weighs about 150000 grams", "7000"]
    assert e["segments"][-1][0] == ". The Lion is about 7000 grams heavier than the Tiger.<|endoftext|>"


def test_a_trace_replays_through_the_agent(tmp_path, monkeypatch):
    """What the data says the runtime hands back is what it does hand back: each kind, run through agent.py."""
    maker, _ = makers(tmp_path, monkeypatch, seed=1)
    for kind in ["heavier", "together", "kilograms", "same_family", "delegate", "graph_save"]:
        for case in ["known", "unknown", "missing"]:
            if case != "known" and kind in ("delegate", "graph_save"):
                continue
            e = maker.make(kind, case if kind != "same_family" or case != "unknown" else "missing")
            # math's script for each question the agent asks it: the calculator's call, then the answer, as episode
            # 21 taught it.
            math = [f" <<calc {q}>>. The answer is {calculate(q)}."
                    for q in re.findall(r"<<ask math: What is ([^?]*)\?>>", show(e))]
            run, box = replay(tmp_path / f"{kind}-{case}", e, math)
            assert [r for _, r, _ in run.calls] == results(e), (kind, case, show(e))
            assert run.final == e["segments"][-1][0].removeprefix(". ").replace("<|endoftext|>", ""), (kind, case)
            for name, content in e["expect"]["files"].items():
                assert box.read(name) == content


def test_the_agent_says_what_the_graph_hasnt_got(tmp_path, monkeypatch):
    maker, _ = makers(tmp_path, monkeypatch)
    for _ in range(20):
        e = maker.make("together", "unknown")
        last = results(e)[-1]
        assert last.startswith("The graph doesn't say how much") and "<<ask math" not in show(e)
        assert e["segments"][-1][0] == f". {last}, so I can't answer that.<|endoftext|>"
        e = maker.make("kilograms", "missing")
        assert results(e)[0].startswith("The graph has no species called") and len(results(e)) == 1


def test_same_family(tmp_path, monkeypatch):
    maker, _ = makers(tmp_path, monkeypatch)
    finals = {maker.make("same_family", "known")["segments"][-1][0] for _ in range(200)}
    assert ". Yes, both are in the family Felidae.<|endoftext|>" in finals
    assert any(f.startswith(". No, ") and "Ursidae" in f for f in finals)


def test_episode_21s_examples_get_the_new_tools():
    old = {"role": "agent", "task": "List the files.", "segments": [["Role: agent\nTools: ask math, ask files, write, "
                                                                     "list\nTask: List the files.\nAnswer:", False]]}
    assert with_episode21_tools(old)["segments"][0][0] == agent.prompt("agent", "List the files.")
    assert old["segments"][0][0].count("ask graph") == 0  # a copy: episode 21's data stays as it was


def test_the_graph_role_is_episode_24s_examples(tmp_path, monkeypatch):
    _, g = makers(tmp_path, monkeypatch)
    e = g.property("status", "Lion", "What is the conservation status of {the}{name}?")
    r = as_role(e)
    assert r["segments"][0][0] == "Role: graph\nTools: cypher\nTask: What is the conservation status of the Lion?\nAnswer:"
    assert r["segments"][1:] == e["segments"][1:]
    assert handed_back(e) == r["expect"]["answer"] == "The Lion is listed as vulnerable"
