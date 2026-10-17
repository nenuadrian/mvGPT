import json
import os
import random
import re
import sys

from agent import NUMBER, prompt
from data import DATA_DIR
from tokenizer import EOT
from tool_data import ASK, ASK_HELD_OUT, WORD, WORD_HELD_OUT, number, web_pages
from tools import calculate

AGENT_DIR = os.path.join(DATA_DIR, "agent")
TRAIN = 60_000
HELD_OUT = 100  # per held-out set

# File names: a word, sometimes with a digit. The held-out sets use words training never sees, so the model has to
# copy a name, not remember one.
NAMES = ("notes shopping prices jan feb mar apr may jun scores todo ideas budget sales totals report results week "
         "plan costs sums list team trip books music garden food rent bills school work gym travel party gifts games "
         "recipes numbers data log points hours miles steps orders stock days club").split()
NAMES_HELD_OUT = "orchard harbor violet summit cobalt meadow lantern pebble willow falcon tundra ember".split()
ITEMS = ("milk eggs bread apples tea rice soap paper pens stamps coffee butter cheese pasta beans onions carrots "
         "lemons honey jam salt sugar flour oats nuts socks gloves batteries tape glue string candles matches "
         "plates cups forks spoons towels sponges brushes paint nails screws bulbs").split()

# How each task is asked: the wordings training sees, then the held-out ones.
FILES_TASKS = {
    "save": (["Save {c} in {f}.", "Write {c} to {f}.", "Put {c} in a file called {f}.",
              "Make a note called {f} that says {c}."], ["Store {c} in {f}."]),
    "read": (["What's in {f}?", "Read {f}.", "Show me {f}.", "What does {f} say?"], ["Open {f} and tell me what it says."]),
    "list": (["Which files are there?", "List the files.", "What files do I have?"], ["Show me my files."]),
    "sum": (["Add up the numbers in {f}.", "What is the total of {f}?", "Sum the numbers in {f}."],
            ["How much do the numbers in {f} come to?"]),
    "sum_save": (["Add up the numbers in {f} and save the total in {g}.", "Put the total of {f} in {g}."],
                 ["Save the sum of {f} to {g}."]),
    "copy": (["Copy {f} to {g}."], ["Make a copy of {f} called {g}."]),
    "sum_many": (["Add up the numbers in {fs}.", "What is the total of {fs}?"], ["How much do {fs} come to altogether?"]),
}
AGENT_TASKS = {
    "math_save": (["Work out {q} and save the answer in {g}.", "Calculate {q} and write it to {g}."],
                  ["Find {q} and store the answer in {g}."]),
    "two_math": (["Work out {q} and {r}, and save both in {g}."], ["Calculate {q} and {r} and store them in {g}."]),
    "word_save": (["{p} Save the answer in {g}."], ["{p} Write the answer to {g}."]),
    "sum_many": FILES_TASKS["sum_many"],
    "totals_save": (["Add up each of {fs} and save the totals in {g}."],
                    ["Work out the total of each of {fs} and store them in {g}."]),
    "save": FILES_TASKS["save"],
    "list": FILES_TASKS["list"],
}
SHARE = {("math", None): 0.25, ("files", "save"): 0.06, ("files", "read"): 0.06, ("files", "list"): 0.03,
         ("files", "sum"): 0.08, ("files", "sum_save"): 0.06, ("files", "copy"): 0.04, ("files", "sum_many"): 0.07,
         ("agent", "math_save"): 0.06, ("agent", "two_math"): 0.05, ("agent", "word_save"): 0.04,
         ("agent", "sum_many"): 0.08, ("agent", "totals_save"): 0.04, ("agent", "save"): 0.02, ("agent", "list"): 0.01,
         ("web", None): 0.05}


def trace(role, task, steps, final, files=None, expect_files=None, answer=None, kind=None):
    """An example: the prompt, then each step's text and call, with the tool's result after it, then the final
    sentence. The prompt and the results are given; the calls and the sentences are what the model learns."""
    segments = [[prompt(role, task), False]]
    for i, (call, result) in enumerate(steps):
        segments += [[(" " if i == 0 else ". ") + f"<<{call}>>", True], [f" = {result}", False]]
    segments.append([(". " if steps else " ") + final + EOT, True])
    return {"role": role, "kind": kind, "task": task, "files": files or {}, "segments": segments,
            "expect": {"answer": answer, "files": expect_files or {}}}


def returned(final):
    # What a subagent hands back, as agent.py's Run.answer works it out: the first number, or the sentence.
    match = NUMBER.search(final)
    return match.group(0) if match else final.rstrip(".")


class Maker:
    def __init__(self, rng, held_out=False):
        self.rng, self.held_out = rng, held_out
        self.names = NAMES_HELD_OUT if held_out else NAMES

    def wording(self, tasks, kind):
        seen, new = tasks[kind]
        return self.rng.choice(new if self.held_out else seen)

    def name(self, taken=()):
        while True:
            name = self.rng.choice(self.names) + (str(self.rng.randint(1, 9)) if self.rng.random() < 0.3 else "") + ".txt"
            if name not in taken:
                return name

    def note(self):
        return ", ".join(self.rng.sample(ITEMS, self.rng.randint(2, 5)))

    def numbers(self, low=3, high=15):
        return [str(self.rng.randint(1, 999)) for _ in range(self.rng.randint(low, high))]

    def others(self, taken):
        # Some files that have nothing to do with the task, as a folder usually has.
        return {self.name(taken): self.note() for _ in range(self.rng.randint(0, 2))}

    def question(self):
        op = self.rng.choice("+-*/")
        a, b = number(self.rng, self.rng.randint(1, 4)), max(number(self.rng, self.rng.randint(1, 3)), 1)
        if op == "/":
            a = b * number(self.rng, self.rng.randint(1, 3))
        return f"{a} {op} {b}"

    # The math role: arithmetic and word problems, with the calculator.
    def math(self, kind=None):
        if self.rng.random() < 0.15:
            # Totals of three or four numbers, which the agent asks for when it adds up totals from several files.
            terms = [str(self.rng.randint(1, 20_000)) for _ in range(self.rng.randint(3, 4))]
            q = " + ".join(terms)
            r = calculate(q)
            return trace("math", f"What is {q}?", [(f"calc {q}", r)], f"The answer is {r}.", answer=r, kind="math")
        if self.rng.random() < 0.6:
            q = self.question()
            ask = self.rng.choice((ASK_HELD_OUT if self.held_out else ASK)[q.split()[1]])
            task, expression, after = ask.format(a=q.split()[0], b=q.split()[2]), q, "The answer is {r}."
        else:
            problems = WORD[-WORD_HELD_OUT:] if self.held_out else WORD[:-WORD_HELD_OUT]
            text, expression, after = self.rng.choice(problems)
            fill = {k: max(number(self.rng, self.rng.randint(1, 3)), 2) for k in "abc"}
            if expression.startswith("{a} - {b}"):
                fill["a"], fill["b"] = max(fill["a"], fill["b"]), min(fill["a"], fill["b"])
            if "/" in expression:
                fill["a"] = fill["b"] * number(self.rng, self.rng.randint(1, 2))
            task, expression, after = text.format(**fill), expression.format(**fill), after.format(r="{r}", **fill)
        r = calculate(expression)
        return trace("math", task, [(f"calc {expression}", r)], after.replace("{r}", r), answer=r, kind="math")

    def solve_math(self, task, expression):
        # What a math subagent writes for a question: the calculator's result, then the answer.
        return calculate(expression)

    # The files role.
    def files(self, kind):
        rng, f = self.rng, self.name()
        wording = self.wording(FILES_TASKS, kind)
        missing = kind in ("read", "sum", "sum_save", "copy") and rng.random() < 0.1
        if kind == "save":
            c = self.note() if rng.random() < 0.7 else ", ".join(self.numbers())
            files = self.others({f})
            return trace("files", wording.format(c=c, f=f), [(f"write {f}: {c}", "ok")], f"I saved it in {f}.",
                         files, {**files, f: c}, kind=kind)
        if kind == "list":
            files = {self.name(): self.note() for _ in range(rng.randint(0, 4))}
            names = ", ".join(sorted(files)) or "none"
            final = f"The files are {names}." if files else "There are no files."
            return trace("files", wording, [("list", names)], final, files, files, names if files else "no files", kind)
        if kind == "sum_many":
            names = []
            while len(names) < rng.randint(2, 4):
                names.append(self.name(names))
            contents = {n: self.numbers() for n in names}  # as long as a file added up on its own can be
            everything = [x for n in names for x in contents[n]]
            total = calculate(" + ".join(everything), longest=10_000)
            fs = ", ".join(names[:-1]) + " and " + names[-1]
            steps = [(f"read {n}", ", ".join(contents[n])) for n in names] + [(f"calc {' + '.join(everything)}", total)]
            files = {n: ", ".join(c) for n, c in contents.items()}
            return trace("files", wording.format(fs=fs), steps, f"The total is {total}.", files, files, total, kind)
        content = ", ".join(self.numbers()) if kind != "read" or rng.random() < 0.3 else self.note()
        if kind == "read" and not missing:
            content = self.note()  # notes, so a subagent's answer is the sentence, not a number in it
        g = self.name({f})
        files = {} if missing else {f: content}
        files.update(self.others({f, g}))
        task = wording.format(f=f, g=g)
        if missing:
            return trace("files", task, [(f"read {f}", "error: no such file")], f"There is no file called {f}.",
                         files, files, "no file", kind + "_missing")
        if kind == "read":
            return trace("files", task, [(f"read {f}", content)], f"{f} says: {content}.", files, files, content, kind)
        if kind == "copy":
            return trace("files", task, [(f"read {f}", content), (f"write {g}: {content}", "ok")],
                         f"I copied {f} to {g}.", files, {**files, g: content}, kind=kind)
        total = calculate(content.replace(", ", " + "), longest=10_000)
        steps = [(f"read {f}", content), (f"calc {content.replace(', ', ' + ')}", total)]
        if kind == "sum":
            return trace("files", task, steps, f"The total is {total}.", files, files, total, kind)
        return trace("files", task, steps + [(f"write {g}: {total}", "ok")], f"The total is {total}, saved in {g}.",
                     files, {**files, g: total}, total, kind)

    # The agent: it delegates to the other two roles, and writes and lists files itself.
    def agent(self, kind):
        rng = self.rng
        wording = self.wording(AGENT_TASKS, kind)
        g = self.name()
        if kind in ("save", "list"):
            example = self.files(kind)
            example.update(role="agent", segments=[[prompt("agent", example["task"]), False]] + example["segments"][1:])
            return example
        if kind == "math_save":
            q = self.question()
            r = calculate(q)
            return trace("agent", wording.format(q=q, g=g), [(f"ask math: What is {q}?", r), (f"write {g}: {r}", "ok")],
                         f"The answer is {r}, saved in {g}.", {}, {g: r}, r, kind)
        if kind == "two_math":
            q, q2 = self.question(), self.question()
            r, r2 = calculate(q), calculate(q2)
            steps = [(f"ask math: What is {q}?", r), (f"ask math: What is {q2}?", r2), (f"write {g}: {r}, {r2}", "ok")]
            return trace("agent", wording.format(q=q, r=q2, g=g), steps,
                         f"The answers are {r} and {r2}, saved in {g}.", {}, {g: f"{r}, {r2}"}, None, kind)
        if kind == "word_save":
            problem = self.math()
            while problem["task"].startswith(("What", "How", "Add", "Sub", "Mul", "Div", "Calc", "Find", "Work", "If")) \
                    or "=" in problem["task"]:
                problem = self.math()  # a word problem, not a sum
            r = problem["expect"]["answer"]
            return trace("agent", wording.format(p=problem["task"], g=g),
                         [(f"ask math: {problem['task']}", r), (f"write {g}: {r}", "ok")],
                         f"The answer is {r}, saved in {g}.", {}, {g: r}, r, kind)
        # Files, delegated: a files subagent adds up each one in a context of its own.
        names = []
        count = rng.randint(1, 4) if kind == "sum_many" else rng.randint(2, 3)
        while len(names) < count:
            names.append(self.name(names + [g]))
        contents = {n: self.numbers() for n in names}
        totals = [calculate(" + ".join(contents[n]), longest=10_000) for n in names]
        files = {n: ", ".join(c) for n, c in contents.items()}
        steps = [(f"ask files: Add up the numbers in {n}.", t) for n, t in zip(names, totals)]
        fs = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        if kind == "sum_many":
            if len(names) == 1:
                return trace("agent", wording.format(fs=fs), steps, f"The total is {totals[0]}.", files, files,
                             totals[0], kind)
            total = calculate(" + ".join(totals))
            steps.append((f"ask math: What is {' + '.join(totals)}?", total))
            return trace("agent", wording.format(fs=fs), steps, f"The total is {total}.", files, files, total, kind)
        saved = ", ".join(totals)
        return trace("agent", wording.format(fs=fs, g=g), steps + [(f"write {g}: {saved}", "ok")],
                     f"The totals are {' and '.join([', '.join(totals[:-1]), totals[-1]])}, saved in {g}.",
                     files, {**files, g: saved}, None, kind)

    def make(self, role, kind):
        return self.math() if role == "math" else self.files(kind) if role == "files" else self.agent(kind)


def build(seed=1337):
    rng = random.Random(seed)
    train_maker = Maker(rng)
    keys, weights = zip(*SHARE.items())
    web = iter(web_pages(0, int(TRAIN * SHARE[("web", None)]) + 10))
    train = []
    for role, kind in rng.choices(keys, weights, k=TRAIN):
        if role == "web":
            train.append({"role": "web", "kind": "web", "task": "", "files": {}, "segments": [[next(web) + EOT, True]],
                          "expect": {}})
        else:
            train.append(train_maker.make(role, kind))
    seen = {(e["role"], e["task"]) for e in train}
    new_maker = Maker(random.Random(seed + 1), held_out=True)

    def held_out(maker, pairs, n=HELD_OUT):
        examples = []
        while len(examples) < n:
            e = maker.make(*rng.choice(pairs))
            if (e["role"], e["task"]) not in seen:
                seen.add((e["role"], e["task"]))
                examples.append(e)
        return examples

    files_kinds = [("files", k) for k in FILES_TASKS]
    agent_kinds = [("agent", k) for k in AGENT_TASKS]
    sets = {"train": train,
            "math": held_out(train_maker, [("math", None)]),
            "files": held_out(train_maker, files_kinds),
            "files_new": held_out(new_maker, files_kinds),
            "agent": held_out(train_maker, agent_kinds),
            "agent_new": held_out(new_maker, agent_kinds)}
    # One long task, four files of 10 to 15 numbers: asked of the agent, it delegates; asked of files, it reads all
    # four into one context. agent_eval.py runs both on these.
    long_maker = Maker(random.Random(seed + 2))
    long_maker.numbers = lambda low=10, high=15: [str(long_maker.rng.randint(1, 999)) for _ in range(rng.randint(10, 15))]
    sets["context"] = []
    while len(sets["context"]) < HELD_OUT:
        names = []
        while len(names) < 4:
            names.append(long_maker.name(names))
        files = {n: ", ".join(long_maker.numbers()) for n in names}
        total = calculate(" + ".join(c.replace(", ", " + ") for c in files.values()), longest=10_000)
        task = f"Add up the numbers in {', '.join(names[:3])} and {names[3]}."
        sets["context"].append({"role": "agent", "kind": "context", "task": task, "files": files, "segments": [],
                                "expect": {"answer": total, "files": files}})
    os.makedirs(AGENT_DIR, exist_ok=True)
    for name, examples in sets.items():
        with open(os.path.join(AGENT_DIR, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in examples)
        roles = {r: sum(e["role"] == r for e in examples) for r in ("math", "files", "agent", "web")}
        print(f"{name + '.jsonl':<18}{len(examples):>7,} examples  " +
              ", ".join(f"{r} {n:,}" for r, n in roles.items() if n))
    return sets


def load(name):
    with open(os.path.join(AGENT_DIR, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def show(example):
    # An example as the model sees it, with what it's given in [brackets].
    return "".join(text if learn else f"[{text}]" for text, learn in example["segments"]).replace(EOT, "")


if __name__ == "__main__":
    sets = build()
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    for role, kind in [("files", "sum_save"), ("agent", "sum_many"), ("agent", "two_math")]:
        examples = [e for e in sets["train"] if e["role"] == role and e["kind"] == kind and len(e["files"]) <= 3]
        print("\n" + re.sub(r" {2,}", " ", show(rng.choice(examples))))
