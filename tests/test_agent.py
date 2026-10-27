import torch

import agent as agent_module
from agent import MAX_CALLS, Agent, Sandbox
from constrain import Names, vocabulary
from model import GPTConfig
from speculative import Draft, Lookup
from tokenizer import BPETokenizer


class Scripted:
    """Stands in for the model: each run of a role writes the next of that role's scripts, a token per step.
    The runtime puts the results in after each call, so the scripts leave them out."""

    def __init__(self, tokenizer, scripts):
        self.tokenizer = tokenizer
        self.config = GPTConfig(vocab_size=len(tokenizer), n_layer=1, block_size=1024)
        self.queue = {role: [tokenizer.encode(t) + [tokenizer.eot_id] for t in texts] for role, texts in scripts.items()}
        self.running = []  # the scripts of the runs in progress, a subagent's on top of its agent's
        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.param])

    def eval(self):
        return self

    def __call__(self, idx, cache=None):
        text = self.tokenizer.decode(idx[0].tolist())
        if text.startswith("Role: "):
            self.running.append(self.queue[text.split("\n")[0][len("Role: "):]].pop(0))
        script = self.running[-1]
        token = script.pop(0)
        if not script:
            self.running.pop()
        logits = torch.full((1, idx.size(1), self.config.vocab_size), -1e9)
        logits[0, -1, token] = 0
        return logits, None


def agent(tmp_path, scripts, files=None):
    tokenizer = BPETokenizer([])  # bytes alone, so each character is a token
    sandbox = Sandbox(str(tmp_path), files)
    return Agent(Scripted(tokenizer, scripts), tokenizer, sandbox), sandbox


def test_the_sandbox_keeps_to_its_folder(tmp_path):
    box = Sandbox(str(tmp_path))
    assert box.write("notes.txt", "milk, eggs") == "ok"
    assert box.read("notes.txt") == "milk, eggs"
    for name in ["../escape.txt", "a/b.txt", "/etc/passwd", "Notes.txt", "notes.py", "x" * 25 + ".txt"]:
        assert box.write(name, "hi") == "error: bad file name", name
        assert box.read(name) == "error: bad file name", name
    assert box.read("missing.txt") == "error: no such file"
    assert box.write("big.txt", "x" * 401) == "error: can't write that"
    assert box.write("call.txt", "<<calc 1 + 1>>") == "error: can't write that"
    for i in range(19):
        box.write(f"f{i}.txt", "1")
    assert box.write("one-more.txt", "1") == "error: too many files"
    assert box.list().startswith("f0.txt, f1.txt")


def test_files_reads_then_adds(tmp_path):
    a, _ = agent(tmp_path, {"files": [" <<read jan.txt>>. <<calc 12 + 30 + 8>>. The total is 50."]},
                 {"jan.txt": "12, 30, 8"})
    run = a.run("files", "Add up the numbers in jan.txt.")
    assert [(c, r) for c, r, _ in run.calls] == [("read jan.txt", "12, 30, 8"), ("calc 12 + 30 + 8", "50")]
    assert run.final == "The total is 50." and run.answer == "50"


def test_a_subagent_works_in_a_context_of_its_own(tmp_path):
    scripts = {"agent": [" <<ask files: Add up the numbers in jan.txt.>>. <<write total.txt: 50>>. Saved 50 in total.txt."],
               "files": [" <<read jan.txt>>. <<calc 12 + 30 + 8>>. The total is 50."]}
    a, box = agent(tmp_path, scripts, {"jan.txt": "12, 30, 8"})
    run = a.run("agent", "Add up the numbers in jan.txt and save the total in total.txt.")
    (ask, answer, sub), (write, ok, _) = run.calls
    assert answer == "50" and sub.role == "files" and ok == "ok"
    assert box.read("total.txt") == "50"
    # The subagent read the file; the agent saw its answer and nothing else.
    assert "12, 30, 8" in sub.text and "12, 30, 8" not in run.text


def test_roles_keep_to_their_tools(tmp_path):
    a, box = agent(tmp_path, {"math": [" <<write x.txt: 1>>. <<read x.txt>>. Done."]})
    run = a.run("math", "Write 1 to x.txt.")
    assert [r for _, r, _ in run.calls] == ["error: no such tool", "error: no such tool"]
    assert box.list() == "none"
    a, _ = agent(tmp_path, {"agent": [" <<ask files: Add up a.txt.>>. Done."], "files": [" <<ask math: 1 + 1>>. No."]})
    sub = a.run("agent", "Add up a.txt.").calls[0][2]
    assert sub.calls[0][1] == "error: no such tool"  # files can't ask, and a subagent is one level deep anyway


def test_a_model_that_loops_is_stopped(tmp_path):
    a, _ = agent(tmp_path, {"math": [" <<calc 1 + 1>>" * (MAX_CALLS + 2) + ". 2."]})
    results = [r for _, r, _ in a.run("math", "1 + 1", max_new_tokens=400).calls]
    assert results[:MAX_CALLS] == ["2"] * MAX_CALLS and results[MAX_CALLS:] == ["error: too many calls"] * 2


def test_a_graph_subagent_hands_back_its_sentence(tmp_path, monkeypatch):
    listed = "Tiger, Lion, Jaguar, Leopard, Cheetah and 3 more"
    monkeypatch.setattr(agent_module, "run_query", lambda cypher, server: listed)
    scripts = {"agent": [" <<ask graph: Which species in Felidae are vulnerable?>>. Done."],
               "graph": [" <<cypher MATCH (s) RETURN s.name>>. The vulnerable species in Felidae are Tiger, Lion, "
                         "Jaguar, Leopard, Cheetah and 3 more."]}
    a, _ = agent(tmp_path, scripts)
    (call, answer, sub), = a.run("agent", "Which species in Felidae are vulnerable?").calls
    # The whole sentence, not the 3 of "and 3 more", which a math or files subagent would hand back.
    assert sub.role == "graph" and sub.calls[0][1].endswith("and 3 more")
    assert answer == "The vulnerable species in Felidae are Tiger, Lion, Jaguar, Leopard, Cheetah and 3 more"


def test_the_graph_role_keeps_to_cypher(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_module, "run_query", lambda cypher, server: "157000")
    scripts = {"agent": [" <<cypher MATCH (s) RETURN s.weight_g>>. <<ask graph: How heavy is the Lion?>>. Done."],
               "graph": [" <<calc 1 + 1>>. <<cypher MATCH (s) RETURN s.weight_g>>. The Lion weighs about 157000 grams."]}
    a, _ = agent(tmp_path, scripts)
    run = a.run("agent", "How heavy is the Lion?")
    assert run.calls[0][1] == "error: no such tool"  # the agent asks the graph; it doesn't query it itself
    sub = run.calls[1][2]
    assert [r for _, r, _ in sub.calls] == ["error: no such tool", "157000"]  # and the graph has no calculator
    assert run.calls[1][1] == "The Lion weighs about 157000 grams"


def test_a_graph_subagent_keeps_to_the_names(tmp_path, monkeypatch):
    queries = []
    monkeypatch.setattr(agent_module, "run_query", lambda cypher, server: queries.append(cypher) or "250000")
    scripts = {"agent": [" <<ask graph: How much does the Okapi weigh?>>. Done."],
               "graph": [" <<cypher MATCH (s:Species {name: 'Okapu'}) RETURN s.weight_g>>. The Okapi weighs about "
                         "250000 grams."]}
    tokenizer = BPETokenizer([])
    names = Names(vocabulary(tokenizer), ["Okapi"])
    a = Agent(Scripted(tokenizer, scripts), tokenizer, Sandbox(str(tmp_path)),
              constrain=lambda role, task: names if role == "graph" else None)
    a.run("agent", "How much does the Okapi weigh?")
    # The model writes the u, and the constraint picks the i, the one token that keeps the name on the list.
    assert queries == ["MATCH (s:Species {name: 'Okapi'}) RETURN s.weight_g"]


def test_the_prompts_list_the_graph():
    assert agent_module.prompt("agent", "Hi.").split("\n")[1] == "Tools: ask math, ask files, ask graph, write, list"
    assert agent_module.prompt("graph", "Hi.").split("\n")[1] == "Tools: cypher"


class Positional:
    """Stands in for the model, for speculative decoding: after its prompt, a context writes its role's script by
    position, so a pass of several tokens writes what a pass of one would, and the script's tools' results line up
    with the runtime's. It keeps the context's tokens on the cache, which a trim cuts back as it does the keys."""

    def __init__(self, tokenizer, scripts):
        self.tokenizer = tokenizer
        self.scripts = {role: tokenizer.encode(text) + [tokenizer.eot_id] for role, text in scripts.items()}
        self.config = GPTConfig(vocab_size=len(tokenizer), n_layer=1, block_size=1024)
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.passes = 0

    def parameters(self):
        return iter([self.param])

    def eval(self):
        return self

    def __call__(self, idx, cache):
        self.passes += 1
        new = idx[0].tolist()
        if cache.position == 0:
            role = self.tokenizer.decode(new).split("\n")[0][len("Role: "):]
            cache.keys, cache.values, cache.script, cache.prompt = [], [], self.scripts[role], len(new)
        logits = torch.full((1, len(new), self.config.vocab_size), -1e9)
        for i in range(len(new)):
            at = cache.position + i + 1 - cache.prompt  # the script's token after this position
            logits[0, i, cache.script[at] if 0 <= at < len(cache.script) else self.tokenizer.eot_id] = 0
        cache.position += len(new)
        return logits, None


def test_speculative_decoding_writes_what_greedy_does(tmp_path):
    scripts = {"agent": " <<ask math: What is 37 * 24?>> = 888. The answer is 888.",
               "math": " <<calc 37 * 24>> = 888. The answer is 888."}
    tokenizer = BPETokenizer([])
    runs = {}
    for name in ("none", "lookup", "draft"):
        model = Positional(tokenizer, scripts)
        draft = {"none": None, "lookup": Lookup(k=4), "draft": Draft(Positional(tokenizer, scripts), k=4)}[name]
        a = Agent(model, tokenizer, Sandbox(str(tmp_path / name)), draft=draft)
        runs[name] = (a.run("agent", "What is 37 * 24?"), model.passes)
    plain, passes = runs["none"]
    for name in ("lookup", "draft"):
        run, fewer = runs[name]
        # The same text and calls, the subagent's too, in fewer of the model's passes.
        assert run.text == plain.text and run.calls[0][2].text == plain.calls[0][2].text
        assert fewer < passes
