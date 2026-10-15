import argparse
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokenizer import detokenize

REPO = os.environ.get("MVGPT_REPO", "nenuadrian/mvGPT")
REVISION = os.environ.get("MVGPT_REVISION", "ep003-sft")


# eq=False: two requests asking for the same thing are still two requests.
@dataclass(eq=False)
class Request:
    ids: list
    max_tokens: int
    temperature: float
    top_k: int
    # Filled in by the model's thread, which sets done when they're ready.
    tokens: list = field(default_factory=list)
    finish_reason: str = "length"
    done: threading.Event = field(default_factory=threading.Event)


@torch.no_grad()
def write(model, r, eos_id):
    x, past = torch.tensor([r.ids]), None
    for _ in range(r.max_tokens):
        out = model(x, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1] / r.temperature
        logits[logits < torch.topk(logits, r.top_k).values[-1]] = -float("inf")
        x = torch.multinomial(torch.softmax(logits, dim=-1), 1)[None]
        if x.item() == eos_id:
            r.finish_reason = "stop"
            return
        r.tokens.append(x.item())


def serve(model, tokenizer, requests):
    # The model's thread: one request at a time, in the order they came.
    while True:
        r = requests.get()
        write(model, r, tokenizer.eos_token_id)
        r.done.set()


def decode(tokenizer, ids):
    # Our word-level tokenizer is saved as WordPiece, whose decode puts spaces around punctuation.
    if type(tokenizer.backend_tokenizer.model).__name__ == "WordPiece":
        return detokenize(tokenizer.convert_ids_to_tokens(ids))
    return tokenizer.decode(ids)


class Server(ThreadingHTTPServer):
    # Connections waiting for the server to accept them. With the default, 5, a crowd gets turned away.
    request_queue_size = 128


def handler(tokenizer, context, requests):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"error": {"message": "not found"}})
            self.reply(200, {"model": f"{REPO}@{REVISION}"})

        def do_POST(self):
            if self.path != "/v1/completions":
                return self.reply(404, {"error": {"message": "not found"}})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = tokenizer(body["prompt"]).input_ids
            # The prompt and what's written after it have to fit in the context window.
            room = context - len(ids)
            if room < 1:
                return self.reply(400, {"error": {"message": f"the prompt is {len(ids)} tokens, of {context}"}})
            r = Request(ids, min(body.get("max_tokens", 100), room), body.get("temperature", 0.8),
                        body.get("top_k", 50))
            # This thread waits while the model's thread writes.
            requests.put(r)
            r.done.wait()
            self.reply(200, {
                "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion", "created": int(time.time()),
                "model": f"{REPO}@{REVISION}",
                "choices": [{"index": 0, "text": decode(tokenizer, r.tokens), "finish_reason": r.finish_reason}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(r.tokens),
                          "total_tokens": len(ids) + len(r.tokens)},
            })

        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass  # a line per request is too many under load

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve the model from the Hub as an OpenAI-style completions API.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION).eval()
    # Requests wait here for the model, which runs on a thread of its own.
    requests = queue.Queue()
    threading.Thread(target=serve, args=(model, tokenizer, requests), daemon=True).start()
    # In Docker, the server has to listen on all interfaces for a published port to reach it.
    host = os.environ.get("HOST", "127.0.0.1")
    server = Server((host, args.port), handler(tokenizer, model.config.max_position_embeddings, requests))
    print(f"serving {REPO}@{REVISION} on http://{host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
