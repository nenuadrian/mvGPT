import argparse
import queue
import shutil
import tempfile
import threading
import time

import gradio as gr

from agent import Agent, Sandbox
from app import THEME
from checkpoint import load_checkpoint
from model import pick_device

# Files each session starts with, so there's something to ask about.
START = {"jan.txt": "120, 45, 300, 18", "feb.txt": "64, 7, 220", "mar.txt": "15, 980, 33, 41, 6",
         "shopping.txt": "milk, eggs, bread"}
EXAMPLES = ["Add up the numbers in jan.txt, feb.txt and mar.txt.",
            "Work out 386 * 29 and 144 * 6, and save both in results.txt.",
            "Add up the numbers in jan.txt and save the total in total.txt.",
            "A café sells 386 cups of coffee a day. How many does it sell in 29 days? Save the answer in cups.txt.",
            "What's in shopping.txt?",
            "Which files are there?"]
lock = threading.Lock()  # one task at a time on the model, whichever session asks


def new_session():
    folder = tempfile.mkdtemp(prefix="mvgpt-agent-")
    Sandbox(folder, START)
    return folder


def files_table(folder):
    box = Sandbox(folder)
    return [[name, box.read(name)] for name in box.names()]


def respond(task, history, folder, model, tokenizer):
    """Runs the task, and shows each step as it happens: the agent's calls, and each subagent as a step of its own,
    with its calls inside it."""
    history = history + [gr.ChatMessage(role="user", content=task)]
    events = queue.Queue()
    agent = Agent(model, tokenizer, Sandbox(folder), on_event=lambda *e: events.put(e))
    done = {}

    def work():
        with lock:
            done["run"] = agent.run("agent", task)
        events.put(("finished",))

    threading.Thread(target=work, daemon=True).start()
    ids, started = {}, {}
    yield history, files_table(folder)
    while True:
        event = events.get()
        kind, run = event[0], event[1] if len(event) > 1 else None
        if kind == "finished":
            break
        if kind == "start" and run.depth > 0:
            ids[id(run)], started[id(run)] = f"sub-{len(ids)}", time.time()
            history.append(gr.ChatMessage(role="assistant", content="",
                                          metadata={"title": f"Subagent · {run.role}: {run.task}",
                                                    "id": ids[id(run)], "status": "pending"}))
        elif kind == "call":
            call, result = event[2], event[3]
            if call.startswith("ask"):
                continue  # the subagent's own step shows it
            parent = ids.get(id(run))
            metadata = {"title": f"🛠 {call}"}  # the brackets would read as an HTML tag
            if parent:
                metadata["parent_id"] = parent
            history.append(gr.ChatMessage(role="assistant", content=f"= {result}", metadata=metadata))
        elif kind == "end" and run.depth > 0:
            for message in history:
                if isinstance(message, gr.ChatMessage) and message.metadata.get("id") == ids[id(run)]:
                    message.content = f"{run.final}\n\nIt handed back **{run.answer}**, from a context of {run.tokens} tokens."
                    message.metadata.update(status="done", duration=round(time.time() - started[id(run)], 1))
        yield history, files_table(folder)
    run = done["run"]
    history.append(gr.ChatMessage(role="assistant",
                                  content=f"{run.final}\n\n*{len(run.calls)} calls, {run.tokens} tokens in the agent's context.*"))
    yield history, files_table(folder)


def main():
    parser = argparse.ArgumentParser(description="The agent in a chat, with a folder of files per session.")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/agent/model.pt")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device())

    with gr.Blocks(title="mvGPT agent") as demo:
        gr.Markdown("# mvGPT, as an agent\nA 110-million-parameter GPT from the series "
                    "[Minimum Viable GPT](https://mvgpt.nenuadrian.com), with a calculator, a folder of files, and "
                    "subagents it hands work to. Each session has a folder of its own.")
        # A folder per session, deleted when the session ends.
        folder = gr.State(new_session, delete_callback=lambda f: shutil.rmtree(f, ignore_errors=True))
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Agent", height=560)
                task = gr.Textbox(label="Task", placeholder=EXAMPLES[0])
                gr.Examples(EXAMPLES, task)
            with gr.Column(scale=2):
                files = gr.Dataframe(headers=["file", "contents"], label="Files", interactive=False, wrap=True)
                reset = gr.Button("Start the files again")

        def ask(text, history, session):
            yield from respond(text, history, session, model, tokenizer)

        def start_again(session):
            shutil.rmtree(session, ignore_errors=True)
            return new_session()

        task.submit(ask, [task, chat, folder], [chat, files]).then(lambda: "", None, task)
        reset.click(start_again, folder, folder).then(files_table, folder, files)
        demo.load(files_table, folder, files)
    demo.launch(server_port=args.port, theme=THEME)


if __name__ == "__main__":
    main()
