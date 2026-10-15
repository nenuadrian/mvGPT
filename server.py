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
def write(model, batch, pad_id, eos_id):
    # Left padding: every prompt ends in the last column, so each row's next token goes after it.
    width = max(len(r.ids) for r in batch)
    x = torch.tensor([[pad_id] * (width - len(r.ids)) + r.ids for r in batch])
    # 1 for a token, 0 for padding, which attention leaves out.
    mask = torch.tensor([[0] * (width - len(r.ids)) + [1] * len(r.ids) for r in batch])
    temperature = torch.tensor([[r.temperature] for r in batch])
    top_k = torch.tensor([[r.top_k] for r in batch])
    past, writing = None, list(batch)
    for _ in range(max(r.max_tokens for r in batch)):
        # A row's positions count its own tokens, not the padding in front of them.
        positions = (mask.cumsum(1) - 1).clamp(min=0)[:, -x.size(1):]
        out = model(x, attention_mask=mask, position_ids=positions, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1] / temperature
        # Each row keeps its own top k: its k-th largest logit, among the largest k of the batch.
        kth = torch.topk(logits, int(top_k.max())).values.gather(1, top_k - 1)
        logits[logits < kth] = -float("inf")
        x = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        mask = torch.cat([mask, torch.ones_like(x)], dim=1)
        for r, token in zip(batch, x[:, 0].tolist()):
            if r not in writing:
                continue  # finished: its row still runs with the others, and what it writes is dropped
            if token == eos_id:
                r.finish_reason = "stop"
                writing.remove(r)
            else:
                r.tokens.append(token)
                if len(r.tokens) == r.max_tokens:
                    writing.remove(r)
        if not writing:
            return


def serve(model, tokenizer, requests, max_batch, wait):
    # The model's thread: it takes the first request, gives others `wait` seconds to join it, up to
    # max_batch, and writes them together. Without the wait, the first of a crowd would go alone.
    while True:
        batch = [requests.get()]
        deadline = time.monotonic() + wait
        while len(batch) < max_batch:
            try:
                batch.append(requests.get(timeout=max(0.0, deadline - time.monotonic())))
            except queue.Empty:
                break
        write(model, batch, tokenizer.pad_token_id, tokenizer.eos_token_id)
        for r in batch:
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
    parser.add_argument("--batch", type=int, default=16, help="the most requests written at once")
    parser.add_argument("--wait", type=float, default=5, help="how long a batch waits to fill, in milliseconds")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION).eval()
    # Requests wait here for the model, which runs on a thread of its own.
    requests = queue.Queue()
    threading.Thread(target=serve, args=(model, tokenizer, requests, args.batch, args.wait / 1000), daemon=True).start()
    # In Docker, the server has to listen on all interfaces for a published port to reach it.
    host = os.environ.get("HOST", "127.0.0.1")
    server = Server((host, args.port), handler(tokenizer, model.config.max_position_embeddings, requests))
    print(f"serving {REPO}@{REVISION} on http://{host}:{args.port}, up to {args.batch} requests at once", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
