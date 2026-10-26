import os
import re
from dataclasses import dataclass, field

import torch

from graph import SERVER, run_query
from model import KVCache
from tools import calculate

# One model, four roles. A role is a line at the top of the prompt, with the tools it may call: the agent
# delegates to the other three, each of which works in a context of its own and hands back its answer.
ROLES = {
    "agent": ["ask math", "ask files", "ask graph", "write", "list"],
    "math": ["calc"],
    "files": ["read", "write", "list", "calc"],
    "graph": ["cypher"],
}
MAX_DEPTH = 1  # a subagent can't ask for subagents of its own
MAX_CALLS = 10  # per context: a model that loops gets stopped
LONGEST_SUM = 1000  # characters in a call to the calculator, for adding up files
# A file name is a short word with .txt, in the session's folder: no slashes, so no way out of it.
NAME = re.compile(r"[a-z0-9_-]{1,24}\.txt")
MAX_FILES, MAX_SIZE = 20, 400
# A number on its own, not one inside a word or a file name like week1.txt.
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)")


def prompt(role, task):
    return f"Role: {role}\nTools: {', '.join(ROLES[role])}\nTask: {task}\nAnswer:"


class Sandbox:
    """The files the model can see: one folder, with limits the model can't change."""

    def __init__(self, folder, files=None):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        for name, content in (files or {}).items():
            self.write(name, content)

    def names(self):
        return sorted(n for n in os.listdir(self.folder) if NAME.fullmatch(n))

    def read(self, name):
        if not NAME.fullmatch(name):
            return "error: bad file name"
        path = os.path.join(self.folder, name)
        if not os.path.exists(path):
            return "error: no such file"
        with open(path) as f:
            return f.read()

    def write(self, name, content):
        if not NAME.fullmatch(name):
            return "error: bad file name"
        if len(content) > MAX_SIZE or "<" in content or ">" in content:
            return "error: can't write that"
        if name not in self.names() and len(self.names()) >= MAX_FILES:
            return "error: too many files"
        with open(os.path.join(self.folder, name), "w") as f:
            f.write(content)
        return "ok"

    def list(self):
        return ", ".join(self.names()) or "none"


@dataclass
class Run:
    role: str
    task: str
    depth: int
    text: str = ""
    calls: list = field(default_factory=list)  # (call, result, the subagent's Run or None)
    tokens: int = 0  # the length of this context: its prompt and what was written in it

    @property
    def final(self):
        """What it wrote after its last call's result: "The total is 50." """
        if not self.calls:
            return self.text.strip()
        tail = self.text.rsplit(">>", 1)[-1]
        result = f" = {self.calls[-1][1]}"
        return (tail[len(result):] if tail.startswith(result) else tail).lstrip(".").strip()

    @property
    def answer(self):
        # What a subagent hands back. The graph hands back its sentence whole: its facts are names as often as
        # numbers, and the 3 in a list's "and 3 more" isn't the answer.
        if self.role == "graph":
            return self.final.rstrip(".")
        # Math and files: the first number of the final sentence, or the sentence with no number.
        number = NUMBER.search(self.final)
        return number.group(0) if number else self.final.rstrip(".")


class Agent:
    def __init__(self, model, tokenizer, sandbox, on_event=None, server=SERVER):
        self.model, self.tokenizer, self.sandbox = model, tokenizer, sandbox
        self.server = server  # where sg serve is listening, for the graph's queries
        # Called with each step, for the app to show: ("start", run), ("call", run, call, result), ("end", run).
        self.on_event = on_event or (lambda *event: None)

    def call(self, run, call):
        """Runs one call from a run, if its role allows it, and returns the text that goes after it."""
        name, _, rest = call.strip().partition(" ")
        tool = name
        if name == "ask":
            tool = "ask " + rest.split(":", 1)[0].strip()
        if tool not in ROLES[run.role]:
            return "error: no such tool", None
        if len(run.calls) >= MAX_CALLS:
            return "error: too many calls", None
        if name == "calc":
            return calculate(rest, longest=LONGEST_SUM), None
        if name == "cypher":
            # Episode 24's query, as plain text, to sg serve, which can only read.
            return run_query(rest.strip(), self.server), None
        if name == "list":
            return self.sandbox.list(), None
        if name == "read":
            return self.sandbox.read(rest.strip()), None
        if name == "write":
            file, _, content = rest.partition(":")
            if not content.strip():
                return "error: nothing to write", None
            return self.sandbox.write(file.strip(), content.strip()), None
        # ask: a subagent, with a fresh context and the tools of its role, which hands back its answer alone.
        if run.depth >= MAX_DEPTH:
            return "error: too deep", None
        if ":" not in rest:
            return "error: no task", None
        role, task = tool.split(" ", 1)[1], rest.split(":", 1)[1].strip()
        sub = self.run(role, task, run.depth + 1)
        return sub.answer, sub

    @torch.no_grad()
    def run(self, role, task, depth=0, max_new_tokens=640):
        model, tokenizer = self.model, self.tokenizer
        model.eval()
        run = Run(role, task, depth)
        self.on_event("start", run)
        device = next(model.parameters()).device
        ids = tokenizer.encode(prompt(role, task))
        cache = KVCache(model.config.n_layer)
        logits, _ = model(torch.tensor([ids], device=device), cache=cache)
        out, opened = [], None
        while len(out) < max_new_tokens and len(ids) + len(out) < model.config.block_size:
            token = logits[0, -1].argmax().item()
            if token == tokenizer.eot_id:
                break
            out.append(token)
            new = [token]
            text = tokenizer.decode(out)
            if opened is None and text.endswith("<<"):
                opened = len(text)
            elif opened is not None and text.endswith(">>"):
                call, opened = text[opened:-2], None
                result, sub = self.call(run, call)
                run.calls.append((call.strip(), result, sub))
                self.on_event("call", run, call.strip(), result)
                tool_ids = tokenizer.encode(f" = {result}")
                out.extend(tool_ids)
                new += tool_ids
            run.text, run.tokens = tokenizer.decode(out), len(ids) + len(out)
            # The newest tokens, the tool's result included, through the model, with the cache for the rest.
            logits, _ = model(torch.tensor([new], device=device), cache=cache)
        self.on_event("end", run)
        return run


def show(run, indent=""):
    """A run as a tree: its calls, with each subagent's run under the call that started it."""
    lines = [f"{indent}{run.role}: {run.task}"]
    for call, result, sub in run.calls:
        lines.append(f"{indent}  <<{call}>> = {result}")
        if sub is not None:
            lines += show(sub, indent + "    ").splitlines()
    lines.append(f"{indent}  -> {run.final}  ({run.tokens} tokens)")
    return "\n".join(lines)


def main():
    import argparse
    import json

    from checkpoint import load_checkpoint
    from model import pick_device

    parser = argparse.ArgumentParser(description="Give the agent a task, with a folder of files to work in.")
    parser.add_argument("checkpoint")
    parser.add_argument("task")
    parser.add_argument("--role", default="agent", choices=list(ROLES))
    parser.add_argument("--folder", default="sandbox", help="the files the model can see")
    parser.add_argument("--files", help='files to start with, as JSON: {"jan.txt": "12, 30, 8"}')
    parser.add_argument("--server", default=SERVER, help="where sg serve is listening, for the graph role")
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())
    sandbox = Sandbox(args.folder, json.loads(args.files) if args.files else None)
    run = Agent(model, tokenizer, sandbox, server=args.server).run(args.role, args.task)
    print(show(run))
    print(f"files: {sandbox.list()}")


if __name__ == "__main__":
    main()
