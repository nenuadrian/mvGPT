import json
import os
import random
import re
import sys

from data import DATA_DIR, WEB_SHARDS, WEB_VALID_PAGES
from tokenizer import EOT
from tools import CLOSE, OPEN, calculate

TOOLS_DIR = os.path.join(DATA_DIR, "tools")
TRAIN = 50_000
HELD_OUT = 200  # per held-out set
SHARE = {"arithmetic": 0.5, "word": 0.25, "numbers": 0.1, "web": 0.15}  # of the training examples

# Ways of asking for each operation. Training sees the first ones, and two more of each are held out, to see
# whether the model calls the calculator for a question worded in a way it hasn't seen.
ASK = {
    "+": ["What is {a} + {b}?", "What is {a} plus {b}?", "Add {a} and {b}.", "How much is {a} plus {b}?",
          "Calculate {a} + {b}.", "What do you get if you add {a} to {b}?", "Find the sum of {a} and {b}.",
          "{a} + {b} = ?", "What's {a} plus {b}?", "Work out {a} + {b}."],
    "-": ["What is {a} - {b}?", "What is {a} minus {b}?", "Subtract {b} from {a}.", "How much is {a} minus {b}?",
          "Calculate {a} - {b}.", "What do you get if you take {b} away from {a}?",
          "Find the difference between {a} and {b}.", "{a} - {b} = ?", "What's {a} minus {b}?", "Work out {a} - {b}."],
    "*": ["What is {a} * {b}?", "What is {a} times {b}?", "Multiply {a} by {b}.", "How much is {a} times {b}?",
          "Calculate {a} * {b}.", "What do you get if you multiply {a} by {b}?", "Find the product of {a} and {b}.",
          "{a} * {b} = ?", "What's {a} times {b}?", "Work out {a} x {b}."],
    "/": ["What is {a} / {b}?", "What is {a} divided by {b}?", "Divide {a} by {b}.", "How much is {a} divided by {b}?",
          "Calculate {a} / {b}.", "What do you get if you divide {a} by {b}?", "Find the quotient of {a} and {b}.",
          "{a} / {b} = ?", "What's {a} divided by {b}?", "Work out {a} / {b}."],
}
ASK_HELD_OUT = {
    "+": ["What is the total of {a} and {b}?", "If you put {a} and {b} together, how much is that?"],
    "-": ["How much is left if you take {b} from {a}?", "What is {b} less than {a}?"],
    "*": ["What is {a} multiplied by {b}?", "How much are {a} lots of {b}?"],
    "/": ["How many times does {b} go into {a}?", "If {a} is shared equally among {b}, how much does each get?"],
}

# Word problems: the question, the call's expression, and the sentence after the result, with {r} for it.
# Division problems are made to come out whole. The last six are held out.
WORD = [
    ("A box holds {a} pencils. How many pencils are in {b} boxes?", "{b} * {a}", "There are {r} pencils in {b} boxes."),
    ("A shop had {a} apples and sold {b} of them. How many apples are left?", "{a} - {b}", "There are {r} apples left."),
    ("Tickets cost {a} dollars each. How much do {b} tickets cost?", "{b} * {a}", "The tickets cost {r} dollars."),
    ("A train travels {a} kilometres each hour. How far does it go in {b} hours?", "{a} * {b}", "It goes {r} kilometres."),
    ("{a} sweets are shared equally between {b} children. How many does each child get?", "{a} / {b}",
     "Each child gets {r} sweets."),
    ("A library has {a} books and buys {b} more. How many books does it have now?", "{a} + {b}", "It has {r} books now."),
    ("Sam saved {a} dollars in March and {b} dollars in April. How much did Sam save in total?", "{a} + {b}",
     "Sam saved {r} dollars in total."),
    ("A farm has {a} rows of trees with {b} trees in each row. How many trees are there?", "{a} * {b}",
     "There are {r} trees."),
    ("A tank holds {a} litres. After {b} litres are used, how much is left?", "{a} - {b}", "There are {r} litres left."),
    ("A school has {a} students in classes of {b}. How many classes are there?", "{a} / {b}", "There are {r} classes."),
    ("A bag of rice weighs {a} grams. How much do {b} bags weigh?", "{b} * {a}", "The bags weigh {r} grams."),
    ("One film is {a} minutes long and another is {b} minutes long. How long are they together?", "{a} + {b}",
     "Together they are {r} minutes long."),
    ("A pack has {a} cards. If {c} packs are opened and {b} cards are lost, how many cards are left?",
     "{c} * {a} - {b}", "There are {r} cards left."),
    ("A café sells {a} cups of coffee a day. How many does it sell in {b} days?", "{a} * {b}", "It sells {r} cups."),
    ("A rope is {a} centimetres long. It is cut into {b} equal pieces. How long is each piece?", "{a} / {b}",
     "Each piece is {r} centimetres long."),
    ("A car park has {a} spaces and {b} cars are parked. How many spaces are free?", "{a} - {b}",
     "There are {r} free spaces."),
    ("Each crate holds {a} bottles. How many bottles are in {b} crates and {c} loose bottles?", "{b} * {a} + {c}",
     "There are {r} bottles."),
    ("A runner runs {a} metres a day for {b} days. How far is that?", "{a} * {b}", "That is {r} metres."),
    ("A town had {a} people. {b} people moved away and {c} moved in. How many people live there now?",
     "{a} - {b} + {c}", "There are {r} people there now."),
    ("A printer prints {a} pages a minute. How many pages does it print in {b} minutes?", "{a} * {b}",
     "It prints {r} pages."),
    # Held out:
    ("A baker makes {a} loaves a day. How many loaves does she make in {b} days?", "{a} * {b}", "She makes {r} loaves."),
    ("A shelf holds {a} jars. There are {b} jars on it. How many more jars fit?", "{a} - {b}", "{r} more jars fit."),
    ("{a} marbles are put into {b} equal bags. How many marbles are in each bag?", "{a} / {b}",
     "There are {r} marbles in each bag."),
    ("A garden has {a} red flowers and {b} yellow flowers. How many flowers are there?", "{a} + {b}",
     "There are {r} flowers."),
    ("A ferry carries {a} people on each trip and makes {b} trips. How many people does it carry?", "{a} * {b}",
     "It carries {r} people."),
    ("A box has {a} chocolates. After eating {b} and buying {c} more, how many are there?", "{a} - {b} + {c}",
     "There are {r} chocolates."),
]
WORD_HELD_OUT = 6

# Questions with numbers in them that need no calculator: the model should answer without a call.
NUMBERS = [("Which is bigger, {a} or {b}?", max, "{r} is bigger."),
           ("Which is smaller, {a} or {b}?", min, "{r} is smaller."),
           ("Which comes first when counting up, {a} or {b}?", min, "{r} comes first."),
           ("Repeat the number {a}.", lambda a, b: a, "{r}.")]
NUMBERS_HELD_OUT = [("Of {a} and {b}, which is larger?", max, "{r} is larger.")]


def number(rng, digits):
    return rng.randint(10 ** (digits - 1) if digits > 1 else 0, 10**digits - 1)


def question(prompt):
    return f"Question: {prompt}\nAnswer:"


def with_call(prompt, expression, after, kind, digits):
    """An example that calls the calculator. Its segments say which text the model learns: not the question,
    and not the calculator's result, which the program running the model writes."""
    r = calculate(expression)
    return {"kind": kind, "prompt": question(prompt), "answer": r, "digits": digits,
            "segments": [[question(prompt), False], [f" {OPEN}{expression}{CLOSE}", True], [f" = {r}", False],
                         [". " + after.replace("{r}", r) + EOT, True]]}


def arithmetic(rng, asks):
    op = rng.choice("+-*/")
    da, db = rng.randint(1, 6), rng.randint(1, 6)
    a, b = number(rng, da), number(rng, db)
    if op == "/":
        b = max(b, 1)
        # Most divisions come out whole, as most questions people ask do; some have decimals.
        if rng.random() < 0.8:
            a = b * number(rng, rng.randint(1, 4))
    return with_call(rng.choice(asks[op]).format(a=a, b=b), f"{a} {op} {b}", "The answer is {r}.", "arithmetic",
                     max(len(str(a)), len(str(b))))


def word_problem(rng, problems):
    text, expression, after = rng.choice(problems)
    # At least 2 of each, so the sentences read right: no "1 cars".
    a, b, c = (max(number(rng, rng.randint(1, 4)), 2) for _ in range(3))
    if expression.startswith("{a} - {b}"):
        a, b = max(a, b), min(a, b)  # a shop can't sell more apples than it has
    if expression == "{c} * {a} - {b}":
        b = min(b, a * c)
    if "/" in expression:
        a = b * number(rng, rng.randint(1, 3))
    fill = {"a": a, "b": b, "c": c}
    return with_call(text.format(**fill), expression.format(**fill), after.format(r="{r}", **fill), "word",
                     max(len(str(n)) for n in (a, b, c)))


def number_question(rng, asks):
    text, pick, after = rng.choice(asks)
    a, b = number(rng, rng.randint(1, 6)), number(rng, rng.randint(1, 6))
    if a == b:
        b += 1
    r = str(pick(a, b))
    prompt = text.format(a=a, b=b)
    return {"kind": "numbers", "prompt": question(prompt), "answer": r, "digits": max(len(str(a)), len(str(b))),
            "segments": [[question(prompt), False], [" " + after.format(r=r) + EOT, True]]}


def web_pages(first, count):
    # Pages from the last file, after the held-out ones: pretraining never read that file.
    import pyarrow.parquet as pq

    path = os.path.join(DATA_DIR, "web", f"{WEB_SHARDS - 1:03d}_00000.parquet")
    pages = pq.read_table(path, columns=["text"]).column(0).to_pylist()[WEB_VALID_PAGES + first:]
    out = []
    for page in pages:
        # The first sentences, up to about 600 characters, ending at a full stop.
        cut = page[:600].rfind(". ")
        if cut > 200 and OPEN.strip() not in page:
            out.append(page[: cut + 1])
        if len(out) == count:
            break
    return out


def build(seed=1337):
    rng = random.Random(seed)
    word_train, word_held_out = WORD[:-WORD_HELD_OUT], WORD[-WORD_HELD_OUT:]
    n = {kind: int(TRAIN * share) for kind, share in SHARE.items()}
    train = [arithmetic(rng, ASK) for _ in range(n["arithmetic"])]
    train += [word_problem(rng, word_train) for _ in range(n["word"])]
    train += [number_question(rng, NUMBERS) for _ in range(n["numbers"])]
    train += [{"kind": "web", "prompt": "", "segments": [[page + EOT, True]]} for page in web_pages(0, n["web"])]
    rng.shuffle(train)
    seen = {e["prompt"] for e in train}

    def held_out(make):
        examples = []
        while len(examples) < HELD_OUT:
            e = make()
            if e["prompt"] not in seen:
                seen.add(e["prompt"])
                examples.append(e)
        return examples

    sets = {
        "train": train,
        # The questions training asks, with numbers it hasn't seen.
        "seen": held_out(lambda: arithmetic(rng, ASK) if rng.random() < 2 / 3 else word_problem(rng, word_train)),
        # Arithmetic asked in ways training never does.
        "phrasing": held_out(lambda: arithmetic(rng, ASK_HELD_OUT)),
        "word": held_out(lambda: word_problem(rng, word_held_out)),
        "numbers": held_out(lambda: number_question(rng, NUMBERS + NUMBERS_HELD_OUT)),
    }
    # The start of web pages, cut in the middle of a sentence: the model should carry on writing, without a call.
    sets["web"] = [{"kind": "web", "prompt": page[: page.rfind(" ", 0, 160)], "segments": []}
                   for page in web_pages(n["web"] + 1000, HELD_OUT)]
    os.makedirs(TOOLS_DIR, exist_ok=True)
    for name, examples in sets.items():
        with open(os.path.join(TOOLS_DIR, f"{name}.jsonl"), "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in examples)
        kinds = {k: sum(e["kind"] == k for e in examples) for k in SHARE if any(e["kind"] == k for e in examples)}
        print(f"{name + '.jsonl':<16}{len(examples):>7,} examples  " + ", ".join(f"{k} {v:,}" for k, v in kinds.items()))
    return sets


def load(name):
    with open(os.path.join(TOOLS_DIR, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def show(example):
    # An example as the model sees it, with the text it doesn't learn in [brackets].
    return "".join(text if learn else f"[{text}]" for text, learn in example["segments"]).replace(EOT, "")


if __name__ == "__main__":
    sets = build()
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    for kind in ("arithmetic", "word", "numbers"):
        print("\n" + show(rng.choice([e for e in sets["train"] if e["kind"] == kind])))
    print("\n" + re.sub(r"\s+", " ", show(rng.choice([e for e in sets["train"] if e["kind"] == "web"])))[:240] + " ...")
