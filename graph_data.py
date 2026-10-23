import argparse
import json
import os
import random
import re
import zlib
from collections import defaultdict

from data import DATA_DIR
from graph import CLOSE, OPEN, SERVER, literal, run_query
from tokenizer import EOT
from tool_data import NUMBERS, NUMBERS_HELD_OUT, number_question, question, show

GRAPH_DIR = os.path.join(DATA_DIR, "graph")
TRAIN = 30_000
HELD_OUT = 200  # per held-out set
SHARE = {"genus": 0.08, "family": 0.13, "order": 0.09, "class": 0.09, "status": 0.14, "weight": 0.11, "taxon": 0.09,
         "list": 0.11, "missing": 0.08, "numbers": 0.08}  # of the training examples

# Animals the graph hasn't got, as it holds mammals and birds alone: asked about them, the model should say so.
OTHER_ANIMALS = ("Komodo dragon, Green sea turtle, Leatherback sea turtle, Saltwater crocodile, Nile crocodile, "
                 "American alligator, Gharial, King cobra, Black mamba, Reticulated python, Green anaconda, "
                 "Gila monster, Marine iguana, Galapagos tortoise, Tuatara, Frilled lizard, Axolotl, "
                 "Chinese giant salamander, Golden poison frog, Cane toad, Olm, Fire salamander, Great white shark, "
                 "Whale shark, Hammerhead shark, Manta ray, Atlantic salmon, Atlantic cod, Bluefin tuna, Coelacanth, "
                 "Clownfish, Electric eel, Red-bellied piranha, Sea lamprey, Monarch butterfly, Honey bee, "
                 "Stag beetle, Desert locust, Emperor dragonfly, Giant squid, Common octopus, Chambered nautilus, "
                 "Box jellyfish, Crown-of-thorns starfish, Horseshoe crab, Coconut crab, American lobster, "
                 "Emperor scorpion, Goliath birdeater, Garden snail, Common earthworm, Medicinal leech").split(", ")

# The facts as smollerGraph loads them, one a line: a node, with its label and properties, or a relationship.
NODE = re.compile(r"\((?P<name>[^:()]+):(?P<label>\w+)(?: \{(?P<properties>.*)\})?\)")
RELATIONSHIP = re.compile(r"\((?P<start>[^()]+)\)-\[:(?P<type>\w+)\]->\((?P<end>[^()]+)\)")
PROPERTY = re.compile(r"(\w+): (?:'([^']*)'|\"([^\"]*)\"|(-?\d+(?:\.\d+)?))")

# How many PARENT_TAXON hops each rank is above a species. 17,603 of Wikidata's 17,614 species have all four, one
# above the other; the rest are left out of the questions about ranks.
RANKS = {"genus": ("Genus", 1), "family": ("Family", 2), "order": ("Order", 3), "class": ("Class", 4)}
CLASSES = {"Aves": "bird", "Mammalia": "mammal"}

# Ways of asking each question: the ones training sees, then the held-out ones. No training wording says "biologists"
# or "call", so the held-out way of asking for a scientific name stays the hard one.
ASK = {
    "genus": (["What genus is {the}{name} in?", "Which genus does {the}{name} belong to?",
               "Name the genus of {the}{name}.", "What is the genus of {the}{name}?",
               "Which genus is {the}{name} part of?", "Tell me the genus of {the}{name}.",
               "In which genus is {the}{name}?"], ["To which genus does {the}{name} belong?"]),
    "family": (["What family is {the}{name} in?", "Which family does {the}{name} belong to?",
                "Name the family of {the}{name}.", "What family of animals is {the}{name} part of?",
                "What is the family of {the}{name}?", "Which family is {the}{name} in?",
                "Tell me the family of {the}{name}.", "In which family is {the}{name}?"],
               ["To which family does {the}{name} belong?"]),
    "order": (["What order is {the}{name} in?", "Which order does {the}{name} belong to?",
               "Name the order of {the}{name}.", "What is the order of {the}{name}?",
               "Which order is {the}{name} part of?", "Tell me the order of {the}{name}.",
               "In which order is {the}{name}?"], ["To which order does {the}{name} belong?"]),
    "class": (["Is {the}{name} a mammal or a bird?", "What class is {the}{name} in?",
               "Is {the}{name} a bird or a mammal?", "Is {the}{name} a bird?", "Is {the}{name} a mammal?",
               "What is the class of {the}{name}?", "Tell me the class of {the}{name}."],
              ["Which class of animals does {the}{name} belong to?"]),
    "status": (["What is the conservation status of {the}{name}?", "How threatened is {the}{name}?",
                "What is the IUCN status of {the}{name}?", "Is {the}{name} threatened?",
                "Is {the}{name} at risk of extinction?", "How does the IUCN list {the}{name}?",
                "Tell me the conservation status of {the}{name}."], ["How endangered is {the}{name}?"]),
    "weight": (["How much does {the}{name} weigh?", "How heavy is {the}{name}?", "What does {the}{name} weigh?",
                "How many grams does {the}{name} weigh?", "Tell me how much {the}{name} weighs.",
                "How heavy is a grown {name}?", "Give the weight of {the}{name}."],
               ["What is the weight of {the}{name}?"]),
    "taxon": (["What is the scientific name of {the}{name}?", "What is {the}{name} known as in Latin?",
               "What is the Latin name of {the}{name}?", "What is the binomial name of {the}{name}?",
               "Give the scientific name of {the}{name}.", "Tell me the Latin name for {the}{name}.",
               "Which scientific name does {the}{name} have?"], ["What do biologists call {the}{name}?"]),
    "list": (["Which species in {group} are {status}?", "Name the {status} species in {group}.",
              "List the {status} species of {group}.", "Which {status} species are in {group}?",
              "What species in {group} are listed as {status}?", "Show me the {status} species in {group}.",
              "Which of the species in {group} are {status}?"], ["What {status} species does {group} have?"]),
}


def read_facts(path):
    """The nodes, each name's label and properties, and each node's parent: Wikidata's facts have one
    relationship type, PARENT_TAXON, from each taxon to the one above it."""
    nodes, parent = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if match := RELATIONSHIP.fullmatch(line):
                parent[match["start"]] = match["end"]
            elif match := NODE.fullmatch(line):
                properties = {key: number_value(number) if number else text or quoted
                              for key, text, quoted, number in PROPERTY.findall(match["properties"] or "")}
                nodes[match["name"]] = (match["label"], properties)
            else:
                raise ValueError(f"not a fact: {line}")
    return nodes, parent


def number_value(text):
    return float(text) if "." in text else int(text)


def held_out(name):
    # A tenth of the names, the same ones each time: training never asks about them.
    return zlib.crc32(name.encode()) % 10 == 0


class Maker:
    def __init__(self, nodes, parent, rng, names_held_out=False, wording_held_out=False, server=SERVER):
        self.nodes, self.rng, self.server, self.wording_held_out = nodes, rng, server, wording_held_out
        self.answers = {}
        self.species = [name for name, (label, _) in nodes.items()
                        if label == "Species" and held_out(name) == names_held_out]
        side = set(self.species)
        # The species with the whole tree above them, a genus, a family, an order and a class.
        self.ranked = []
        # Each family's and order's species' statuses, so most questions about a group's species have answers.
        statuses = defaultdict(set)
        for name, (label, properties) in nodes.items():
            if label != "Species":
                continue
            above, node = [], name
            while node in parent:
                node = parent[node]
                above.append(nodes[node][0])
            if above == [label for label, _ in RANKS.values()]:
                if name in side:
                    self.ranked.append(name)
                family = parent[parent[name]]
                statuses[family].add(properties["status"])
                statuses[parent[family]].add(properties["status"])
        self.weighed = [name for name in self.species if "weight_g" in nodes[name][1]]
        # Two species have no scientific name on Wikidata, and some go by theirs alone.
        self.latin = [name for name in self.species if nodes[name][1].get("taxon", name) != name]
        self.common = [name for name in self.latin if " " in name]
        self.others = [name for name in OTHER_ANIMALS if name not in nodes and held_out(name) == names_held_out]
        # Families and orders are held out the way species are.
        self.statuses = {group: sorted(s) for group, s in statuses.items() if held_out(group) == names_held_out}
        self.groups = sorted(self.statuses)
        self.all_statuses = sorted({s for group in statuses.values() for s in group})

    def ask(self, cypher):
        # sg serve answers each query once: the same question about the same species gets the same answer.
        if cypher not in self.answers:
            self.answers[cypher] = run_query(cypher, self.server)
        return self.answers[cypher]

    def wording(self, kind):
        seen, new = ASK[kind]
        return self.rng.choice(new if self.wording_held_out else seen)

    def articles(self, name):
        # "the Red fox", but a species Wikidata names by its scientific name goes without one: "Emblema pictum".
        the = "" if name in self.nodes and self.nodes[name][1].get("taxon") == name else "the "
        return {"name": name, "the": the, "The": the.capitalize()}

    def trace(self, kind, prompt, cypher, final, name=None):
        """An example that queries the graph. Its segments say which text the model learns: not the question, and
        not sg serve's answer, which the program running the model writes."""
        result = self.ask(cypher)
        if result.startswith("error"):
            raise ValueError(f"{cypher}: {result}")
        if result == "nothing" and name is not None:
            # A species the graph hasn't got: the model says so, rather than make an answer up.
            kind, final = "missing", f"The graph has no species called {name}."
        else:
            final = final(result)
        return {"kind": kind, "prompt": question(prompt), "cypher": cypher, "result": result,
                "segments": [[question(prompt), False], [f" {OPEN}{cypher}{CLOSE}", True], [f" = {result}", False],
                             [". " + final + EOT, True]]}

    def made_up(self):
        # A name made of two real ones, which the graph hasn't got: "Westland Shrew".
        while True:
            first, last = self.rng.sample(self.common, 2)
            name = f"{first.rsplit(' ', 1)[0]} {last.rsplit(' ', 1)[1]}"
            if name not in self.nodes:
                return name

    def missing(self):
        # Half of them animals that are neither mammals nor birds, half made up.
        name = self.rng.choice(self.others) if self.rng.random() < 0.5 else self.made_up()
        kind = self.rng.choice([*RANKS, "status", "weight", "taxon"])
        return self.rank(kind, name) if kind in RANKS else self.property(kind, name)

    def rank(self, kind, name=None):
        name = name or self.rng.choice(self.ranked)
        label, hops = RANKS[kind]
        v = label[0].lower()
        cypher = f"MATCH (s:Species {{name: {literal(name)}}})-[:PARENT_TAXON*{hops}]->({v}:{label}) RETURN {v}.name"
        a = self.articles(name)
        if kind == "class":
            final = lambda r: f"{a['The']}{name} is a {CLASSES[r]}, in the class {r}."
        else:
            final = lambda r: f"{a['The']}{name} is in the {kind} {r}."
        return self.trace(kind, self.wording(kind).format(**a), cypher, final, name)

    def property(self, kind, name=None):
        if name is None and kind == "weight":
            # Most species have no weight on Wikidata: some questions about those, so the model says it doesn't know.
            name = self.rng.choice(self.weighed if self.rng.random() < 0.7 else self.species)
        elif name is None:
            name = self.rng.choice(self.latin if kind == "taxon" else self.species)
        key = {"status": "status", "weight": "weight_g", "taxon": "taxon"}[kind]
        cypher = f"MATCH (s:Species {{name: {literal(name)}}}) RETURN s.{key}"
        a = self.articles(name)
        the, The = a["the"], a["The"]
        final = {
            "status": lambda r: f"{The}{name} is listed as {r}.",
            "weight": lambda r: (f"The graph doesn't say how much {the}{name} weighs." if r == "unknown"
                                 else f"{The}{name} weighs about {r} grams."),
            "taxon": lambda r: f"The scientific name of {the}{name} is {r}.",
        }[kind]
        return self.trace(kind, self.wording(kind).format(**a), cypher, final, name)

    def listing(self):
        group = self.rng.choice(self.groups)
        label = self.nodes[group][0]
        hops, v = RANKS[label.lower()][1], label[0].lower()
        # Mostly a status some of its species have; sometimes any, which may find none.
        status = self.rng.choice(self.statuses[group] if self.rng.random() < 0.85 else self.all_statuses)
        cypher = (f"MATCH (s:Species)-[:PARENT_TAXON*{hops}]->({v}:{label} {{name: {literal(group)}}}) "
                  f"WHERE s.status = {literal(status)} RETURN s.name")

        def final(r):
            if r == "nothing":
                return f"None of the species in {group} is {status}."
            if "," not in r:
                return f"The only {status} species in {group} is {r}."
            return f"The {status} species in {group} are {r}."

        return self.trace("list", self.wording("list").format(group=group, status=status), cypher, final)

    def make(self, kind):
        if kind in RANKS:
            return self.rank(kind)
        if kind == "list":
            return self.listing()
        if kind == "missing":
            return self.missing()
        if kind == "numbers":
            return number_question(self.rng, NUMBERS_HELD_OUT if self.wording_held_out else NUMBERS)
        return self.property(kind)


def build(facts, server=SERVER, train_size=TRAIN, seed=1337):
    rng = random.Random(seed)
    nodes, parent = read_facts(facts)
    kinds = list(SHARE)
    train_maker = Maker(nodes, parent, rng, server=server)
    train = [train_maker.make(kind) for kind in kinds for _ in range(int(train_size * SHARE[kind]))]
    rng.shuffle(train)
    seen = {e["prompt"] for e in train}
    # The held-out sets draw from a generator of their own, so changing what training asks doesn't change them,
    # and two models trained on different data are measured on the same questions.
    held_rng = random.Random(seed + 1)

    def held(maker, kinds):
        examples = []
        while len(examples) < HELD_OUT:
            e = maker.make(held_rng.choice(kinds))
            if e["prompt"] not in seen:
                seen.add(e["prompt"])
                examples.append(e)
        return examples

    graph_kinds = [k for k in kinds if k not in ("missing", "numbers")]
    sets = {
        "train": train,
        # The questions training asks, about species and groups it never names.
        "seen": held(Maker(nodes, parent, held_rng, names_held_out=True, server=server), graph_kinds),
        # Asked in ways training never does, about those species.
        "phrasing": held(Maker(nodes, parent, held_rng, True, True, server), graph_kinds),
        # About animals the graph hasn't got, which training never names either.
        "missing": held(Maker(nodes, parent, held_rng, names_held_out=True, server=server), ["missing"]),
        # Questions the graph has nothing to do with: the model should answer without a call.
        "numbers": held(Maker(nodes, parent, held_rng, server=server), ["numbers"]),
    }
    os.makedirs(GRAPH_DIR, exist_ok=True)
    for name, examples in sets.items():
        with open(os.path.join(GRAPH_DIR, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in examples)
        counts = {k: sum(e["kind"] == k for e in examples) for k in kinds if any(e["kind"] == k for e in examples)}
        print(f"{name + '.jsonl':<16}{len(examples):>7,} examples  " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    queries = len(train_maker.answers)
    print(f"{queries:,} queries to sg serve for training, "
          f"{sum(r == 'nothing' for r in train_maker.answers.values()):,} of them found nothing")
    return sets


def load(name):
    with open(os.path.join(GRAPH_DIR, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Questions about animals, answered by querying smollerGraph.")
    parser.add_argument("facts", help="smollerGraph's data/wikidata.txt, which sg serve has loaded")
    parser.add_argument("--server", default=SERVER, help="where sg serve is listening")
    parser.add_argument("--train", type=int, default=TRAIN, help="training examples")
    parser.add_argument("--seed", type=int, default=0, help="which examples to show")
    args = parser.parse_args()
    sets = build(args.facts, args.server, args.train)
    rng = random.Random(args.seed)
    for kind in SHARE:
        print("\n" + show(rng.choice([e for e in sets["train"] if e["kind"] == kind])))
