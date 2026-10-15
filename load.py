import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

# What players in a game could ask a storyteller for: words from nanoMMO's world that the model knows.
WORDS = ["sword", "gate", "skeleton", "meadow", "ash", "knight", "dragon", "castle", "forest", "bridge", "village",
         "king", "queen", "shield", "brave", "hero", "monster", "bones", "fire", "river", "island", "gold"]


def client(url, requests, seed, max_tokens, results, errors):
    # One player: a request, the wait for its story, then the next.
    rng = random.Random(seed)
    for _ in range(requests):
        prompt = f"Words: {', '.join(rng.sample(WORDS, 3))}\nStory:"
        body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
        start = time.perf_counter()
        try:
            request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(request) as response:
                tokens = json.load(response)["usage"]["completion_tokens"]
        except (urllib.error.URLError, ConnectionError):
            errors.append(seed)
            continue
        results.append((time.perf_counter() - start, tokens))


def run(url, clients, requests, max_tokens):
    results, errors = [], []
    threads = [threading.Thread(target=client, args=(url, requests, seed, max_tokens, results, errors))
               for seed in range(clients)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seconds = time.perf_counter() - start
    if not results:
        raise SystemExit(f"every request failed: is the server running at {url}?")
    latencies = sorted(latency for latency, _ in results)
    return {"clients": clients, "requests": len(results), "errors": len(errors), "seconds": round(seconds, 2),
            "tokens_per_second": round(sum(tokens for _, tokens in results) / seconds),
            "median_ms": round(statistics.median(latencies) * 1000),
            # The latency 95% of requests beat: what the unluckiest players wait.
            "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))] * 1000)}


def main():
    parser = argparse.ArgumentParser(description="Send a server many requests at once, and measure how it keeps up.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 4, 16, 64], help="how many at once, in turn")
    parser.add_argument("--requests", type=int, default=4, help="requests each client sends, one after another")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--out", help="a file to write the measurements to, as JSON")
    args = parser.parse_args()

    run(args.url, 1, 1, 10)  # one short request, to warm up
    print(f"{'clients':<10}{'requests':>10}{'errors':>8}{'tokens/s':>10}{'median':>10}{'p95':>10}")
    rows = []
    for clients in args.clients:
        row = run(args.url, clients, args.requests, args.max_tokens)
        rows.append(row)
        print(f"{clients:<10}{row['requests']:>10}{row['errors']:>8}{row['tokens_per_second']:>10}"
              f"{row['median_ms']:>8}ms{row['p95_ms']:>8}ms")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
