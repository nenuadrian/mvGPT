import argparse
import json
import os
import random
from collections import defaultdict

from agent import prompt
from agent_data import Maker as FilesMaker
from agent_data import load as load_episode21
from agent_data import show, trace
from data import DATA_DIR
from graph import SERVER
from graph_data import SHARE as GRAPH_SHARE
from graph_data import Maker as GraphMaker
from graph_data import read_facts
from tokenizer import EOT
from tools import calculate

GRAPH_AGENT_DIR = os.path.join(DATA_DIR, "graph-agent")
TRAIN = 45_000
HELD_OUT = 200  # per held-out set
# Of the training examples: episode 21's, so the agent keeps its three roles; the graph role, episode 24's questions
# under its role line; and the agent's new tasks, which ask the graph and then, most of them, the calculator.
SHARE = {"episode21": 0.45, "graph": 0.25, "heavier": 0.07, "together": 0.04, "kilograms": 0.04,
         "same_family": 0.05, "delegate": 0.06, "graph_save": 0.04}
GRAPH_KINDS = [k for k in GRAPH_SHARE if k != "numbers"]  # the graph role answers from the graph, or says it can't
# Of the tasks about weights, how many name a species the graph has no weight for, or one it hasn't got at all.
UNKNOWN, MISSING = 0.15, 0.05

# How the agent asks its graph subagent for a fact: one way each, in episode 24's training wordings.
ASK_WEIGHT = "How much does {the}{name} weigh?"
ASK_FAMILY = "What family is {the}{name} in?"

# How each task is asked: the wordings training sees, then the held-out one. {a} and {b} are "the Lion" or a
# scientific name, which goes without one. Episode 21's held-out wordings ("Write the answer to", "store") stay out.
TASKS = {
    "heavier": (["How much heavier is {a} than {b}?", "How many grams heavier is {a} than {b}?",
                 "How much more does {a} weigh than {b}?", "By how much does {a} outweigh {b}?",
                 "What is the difference in weight between {a} and {b}?"],
                ["How much more weight does {a} carry than {b}?"]),
    "together": (["How much do {a} and {b} weigh together?", "What do {a} and {b} weigh altogether?",
                  "What is the combined weight of {a} and {b}?", "How heavy are {a} and {b} together?"],
                 ["What would {a} and {b} weigh on one scale?"]),
    "kilograms": (["How many kilograms does {a} weigh?", "What is the weight of {a} in kilograms?",
                   "How heavy is {a} in kilograms?", "Give the weight of {a} in kilograms."],
                  ["How much does {a} weigh in kg?"]),
    "same_family": (["Are {a} and {b} in the same family?", "Do {a} and {b} belong to the same family?",
                     "Is {a} in the same family as {b}?", "Are {a} and {b} part of the same family?"],
                    ["Do {a} and {b} share a family?"]),
    "graph_save": (["{q} Save the answer in {g}.", "{q} Put the answer in {g}."], ["{q} Keep the answer in {g}."]),
}
AGENT_KINDS = [*TASKS, "delegate"]


def question_text(example):
    # The question of one of graph_data.py's examples, without "Question: " and "\nAnswer:".
    return example["prompt"].removeprefix("Question: ").removesuffix("\nAnswer:")


def handed_back(example):
    # What a graph subagent hands back, as agent.py's Run.answer works it out: its final sentence, without the stop.
    return example["segments"][-1][0].removeprefix(". ").replace(EOT, "").strip().rstrip(".")


def as_role(example):
    """One of episode 24's examples under the graph role's line, in place of "Question:": the same query, sg's
    answer and sentence."""
    task = question_text(example)
    return {"role": "graph", "kind": example["kind"], "task": task, "files": {},
            "segments": [[prompt("graph", task), False]] + [list(s) for s in example["segments"][1:]],
            "expect": {"answer": handed_back(example), "files": {}}}


def with_episode21_tools(example):
    # One of episode 21's examples, with the role's line of tools as it is now: the agent's has `ask graph`.
    example = json.loads(json.dumps(example))
    if example["role"] != "web":
        example["segments"][0][0] = prompt(example["role"], example["task"])
    return example


class Maker:
    """The agent's new tasks. Each fact comes from a graph subagent, whose sentence is built from sg serve's answer,
    and each sum from the calculator, so the traces can't disagree with the database. Each example keeps its graph
    subagents' runs, as the graph role would write them, under "subagents"."""

    def __init__(self, graph, parent, rng, held_out=False):
        self.graph, self.rng, self.held_out = graph, rng, held_out
        self.files = FilesMaker(rng, held_out)  # file names, from episode 21's, the held-out ones for held-out sets
        self.unweighed = [name for name in graph.species if "weight_g" not in graph.nodes[name][1]]
        # Each family's species, of those with the whole tree above them, for pairs in the same family.
        self.family = {name: parent[parent[name]] for name in graph.ranked}
        self.in_family = defaultdict(list)
        for name, family in self.family.items():
            self.in_family[family].append(name)
        self.paired = [name for name in graph.ranked if len(self.in_family[self.family[name]]) > 1]

    def wording(self, kind):
        seen, new = TASKS[kind]
        return self.rng.choice(new if self.held_out else seen)

    def the(self, name, capital=False):
        a = self.graph.articles(name)
        return (a["The"] if capital else a["the"]) + name

    def absent(self):
        # A name the graph hasn't got: an animal that's neither a mammal nor a bird, or one made of two real names.
        return self.rng.choice(self.graph.others) if self.rng.random() < 0.5 else self.graph.made_up()

    def fact(self, kind, name):
        """A graph subagent's run for one fact, asked one way every time: the agent's call, what comes back, sg's
        own answer ("157000", "unknown", "nothing") and the subagent's example."""
        if kind == "weight":
            sub = self.graph.property("weight", name, ASK_WEIGHT)
        else:
            sub = self.graph.rank("family", name, ASK_FAMILY)
        return f"ask graph: {question_text(sub)}", handed_back(sub), sub["result"], as_role(sub)

    def weights(self, kind, case=None):
        """heavier, together and kilograms: one or two weights from the graph, then the calculator. case is "known",
        "unknown" (a species without a weight) or "missing" (one the graph hasn't got), or random."""
        rng = self.rng
        if case is None:
            r = rng.random()
            case = "missing" if r < MISSING else "unknown" if r < MISSING + UNKNOWN else "known"
        names = rng.sample(self.graph.weighed, 1 if kind == "kilograms" else 2)
        if case != "known":
            names[rng.randrange(len(names))] = self.absent() if case == "missing" else rng.choice(self.unweighed)
        facts = [self.fact("weight", name) for name in names]
        if kind == "heavier" and case == "known":
            if facts[0][2] == facts[1][2]:
                return self.weights(kind, case)  # neither is heavier
            if float(facts[0][2]) < float(facts[1][2]):
                names, facts = names[::-1], facts[::-1]  # the heavier one first, as the question asks
        a, b = self.the(names[0]), self.the(names[-1])
        task = self.wording(kind).format(a=a, b=b)
        steps, subagents = [], []
        for call, answer, result, sub in facts:
            steps.append((call, answer))
            subagents.append(sub)
            if result in ("unknown", "nothing"):
                # The agent stops at the first weight it can't get, and says why, rather than make one up.
                return self.example(task, steps, f"{answer}, so I can't answer that.", answer, kind, subagents, case)
        w = [result for _, _, result, _ in facts]
        if kind == "heavier":
            r = calculate(f"{w[0]} - {w[1]}")
            steps.append((f"ask math: What is {w[0]} - {w[1]}?", r))
            final = f"{self.the(names[0], True)} is about {r} grams heavier than {b}."
        elif kind == "together":
            r = calculate(f"{w[0]} + {w[1]}")
            steps.append((f"ask math: What is {w[0]} + {w[1]}?", r))
            final = f"{self.the(names[0], True)} and {b} weigh about {r} grams together."
        else:
            r = calculate(f"{w[0]} / 1000")
            steps.append((f"ask math: What is {w[0]} / 1000?", r))
            final = f"{self.the(names[0], True)} weighs about {r} kilograms."
        return self.example(task, steps, final, r, kind, subagents, case)

    def same_family(self, case=None):
        """Two families from the graph, and the agent compares them itself: no calculator."""
        rng = self.rng
        if case is None:
            case = "missing" if rng.random() < MISSING else "known"
        a = rng.choice(self.paired)
        if rng.random() < 0.5:
            b = rng.choice([name for name in self.in_family[self.family[a]] if name != a])
        else:
            b = rng.choice(self.graph.ranked)
            while b == a:
                b = rng.choice(self.graph.ranked)
        names = [a, b]
        if case == "missing":
            names[rng.randrange(2)] = self.absent()
        task = self.wording("same_family").format(a=self.the(names[0]), b=self.the(names[1]))
        facts = [self.fact("family", name) for name in names]
        steps, subagents = [], []
        for call, answer, result, sub in facts:
            steps.append((call, answer))
            subagents.append(sub)
            if result == "nothing":
                return self.example(task, steps, f"{answer}, so I can't answer that.", answer, "same_family",
                                    subagents, case)
        fa, fb = facts[0][2], facts[1][2]
        if fa == fb:
            final = f"Yes, both are in the family {fa}."
        else:
            final = f"No, {self.the(names[0])} is in {fa} and {self.the(names[1])} is in {fb}."
        return self.example(task, steps, final, final.rstrip("."), "same_family", subagents, case)

    def delegate(self):
        """Any of episode 24's questions, handed to the graph as it's asked, as the agent hands word problems to
        math: lists and names the graph hasn't got included."""
        sub = self.graph.make(self.rng.choice(GRAPH_KINDS))
        q, answer = question_text(sub), handed_back(sub)
        return self.example(q, [(f"ask graph: {q}", answer)], answer + ".", answer, "delegate", [as_role(sub)])

    def graph_save(self):
        """A fact about a species the graph has, from the graph, saved to a file with episode 21's write."""
        sub = self.graph.make(self.rng.choice(["genus", "family", "order", "class", "status", "weight", "taxon"]))
        q, answer = question_text(sub), handed_back(sub)
        g = self.files.name()
        steps = [(f"ask graph: {q}", answer), (f"write {g}: {answer}", "ok")]
        return self.example(self.wording("graph_save").format(q=q, g=g), steps, f"{answer}. I saved it in {g}.",
                            answer, "graph_save", [as_role(sub)], files={g: answer})

    def example(self, task, steps, final, answer, kind, subagents, case="known", files=None):
        e = trace("agent", task, steps, final, {}, files or {}, answer, kind)
        e["case"], e["subagents"] = case, subagents
        return e

    def make(self, kind, case=None):
        if kind in ("heavier", "together", "kilograms"):
            return self.weights(kind, case)
        if kind == "same_family":
            return self.same_family(case)
        return self.delegate() if kind == "delegate" else self.graph_save()


def build(facts, server=SERVER, train_size=TRAIN, seed=1337):
    rng = random.Random(seed)
    nodes, parent = read_facts(facts)
    graph = GraphMaker(nodes, parent, rng, server=server)
    maker = Maker(graph, parent, rng)
    # Episode 21's examples, as agent_data.py wrote them (run it first), a sample of them.
    episode21 = load_episode21("train")
    train = [with_episode21_tools(e) for e in rng.sample(episode21, int(train_size * SHARE["episode21"]))]
    total = sum(GRAPH_SHARE[k] for k in GRAPH_KINDS)
    for kind in GRAPH_KINDS:
        n = int(train_size * SHARE["graph"] * GRAPH_SHARE[kind] / total)
        train += [as_role(graph.make(kind)) for _ in range(n)]
    for kind in AGENT_KINDS:
        train += [maker.make(kind) for _ in range(int(train_size * SHARE[kind]))]
    rng.shuffle(train)
    seen = {(e["role"], e["task"]) for e in train}
    # The held-out sets draw from a generator of their own, about species, families and file names training never
    # names, so changing what training asks doesn't change them.
    held_rng = random.Random(seed + 1)

    def held(held_out_wordings, kinds, case):
        held_graph = GraphMaker(nodes, parent, held_rng, names_held_out=True, wording_held_out=held_out_wordings,
                                server=server)
        held_maker = Maker(held_graph, parent, held_rng, held_out=held_out_wordings)
        examples = []
        while len(examples) < HELD_OUT:
            kind = held_rng.choice(kinds)
            e = held_maker.make(kind, case if kind != "same_family" or case != "unknown" else "missing")
            if ("agent", e["task"]) not in seen:
                seen.add(("agent", e["task"]))
                examples.append(e)
        return examples

    sets = {
        "train": train,
        # The tasks training asks, about species it never names.
        "seen": held(False, AGENT_KINDS, "known"),
        # Asked in ways training never does.
        "phrasing": held(True, AGENT_KINDS, "known"),
        # A species without a weight, or one the graph hasn't got: the agent should say it can't.
        "unknown": held(False, ["heavier", "together", "kilograms", "same_family"], "unknown"),
    }
    os.makedirs(GRAPH_AGENT_DIR, exist_ok=True)
    for name, examples in sets.items():
        with open(os.path.join(GRAPH_AGENT_DIR, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in examples)
        counts = defaultdict(int)
        for e in examples:
            counts[e["role"] if e["role"] != "agent" or e["kind"] not in AGENT_KINDS else e["kind"]] += 1
        print(f"{name + '.jsonl':<16}{len(examples):>7,} examples  " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    print(f"{len(graph.answers):,} queries to sg serve for training, "
          f"{sum(r == 'nothing' for r in graph.answers.values()):,} of them found nothing")
    return sets


def load(name):
    with open(os.path.join(GRAPH_AGENT_DIR, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The agent's tasks that ask smollerGraph, with episode 21's.")
    parser.add_argument("facts", help="smollerGraph's data/wikidata.txt, which sg serve has loaded")
    parser.add_argument("--server", default=SERVER, help="where sg serve is listening")
    parser.add_argument("--train", type=int, default=TRAIN, help="training examples")
    parser.add_argument("--seed", type=int, default=0, help="which examples to show")
    args = parser.parse_args()
    sets = build(args.facts, args.server, args.train)
    rng = random.Random(args.seed)
    for kind in AGENT_KINDS:
        e = rng.choice([e for e in sets["train"] if e["kind"] == kind and e.get("case") == "known"])
        print("\n" + show(e))
        for sub in e["subagents"]:
            print("    " + show(sub).replace("\n", "\n    "))
    e = rng.choice([e for e in sets["train"] if e.get("case") == "unknown"])
    print("\n" + show(e))
