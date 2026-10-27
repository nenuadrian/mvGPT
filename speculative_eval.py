import argparse
import json
import os
import time

from agent import Agent
from agent_eval import attempt
from checkpoint import load_checkpoint
from constrain import task_words
from graph import SERVER
from graph_agent_data import load
from hellaswag import RESULTS
from model import pick_device
from speculative import Draft, Lookup

SETS = ["seen", "phrasing", "unknown"]  # episode 25's held-out tasks


class Counted:
    """The model, counting its forward passes."""

    def __init__(self, model):
        self.model, self.config, self.passes = model, model.config, 0

    def __call__(self, *args, **kwargs):
        self.passes += 1
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()


def main():
    parser = argparse.ArgumentParser(description="The agent's tasks with each draft, against greedy decoding alone.")
    parser.add_argument("checkpoint")
    parser.add_argument("draft", help="the small model's checkpoint")
    parser.add_argument("-k", type=int, default=4, help="tokens the drafts guess in each pass")
    parser.add_argument("--limit", type=int, help="only the first tasks of each set")
    parser.add_argument("--server", default=SERVER, help="where sg serve is listening")
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    small = load_checkpoint(args.draft, device)[0].eval()
    constrain = task_words(tokenizer)  # names copied from the task, as in episode 26
    examples = [e for set_name in SETS for e in load(set_name)[: args.limit]]
    drafts = {"none": lambda: None, "lookup": lambda: Lookup(args.k), "draft": lambda: Draft(small, args.k)}
    results, greedy = {"model": os.path.basename(os.path.dirname(args.checkpoint)), "k": args.k, "drafts": {}}, None
    print(f"{len(examples)} tasks, {args.k} guesses a pass\n{'draft':<8}{'right':>8}{'same text':>11}{'passes':>9}"
          f"{'kept':>8}{'seconds':>9}{'faster':>8}")
    for name, make in drafts.items():
        counted, agents = Counted(model), []

        def agent_for(sandbox):
            agents.append(Agent(counted, tokenizer, sandbox, server=args.server, constrain=constrain, draft=make()))
            return agents[-1]

        start = time.perf_counter()
        rows = [attempt(agent_for, e) for e in examples]
        seconds = (time.perf_counter() - start) / len(rows)
        guessed, kept = sum(a.guessed for a in agents), sum(a.kept for a in agents)
        traces = [r["trace"] for r in rows]
        greedy = greedy or {"traces": traces, "seconds": seconds}
        d = {"right": sum(r["right"] for r in rows) / len(rows), "seconds": seconds, "passes": counted.passes / len(rows),
             "guessed": guessed, "kept": kept / guessed if guessed else 0.0,
             "same": sum(t == g for t, g in zip(traces, greedy["traces"])) / len(rows),
             "faster": greedy["seconds"] / seconds}
        results["drafts"][name] = d
        print(f"{name:<8}{d['right']:>8.1%}{d['same']:>11.1%}{d['passes']:>9.1f}{d['kept']:>8.1%}{d['seconds']:>9.2f}"
              f"{d['faster']:>7.2f}x", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"speculative-{results['model']}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
