"""How long a long prefill holds up everything else.

While one long prompt prefills: a stream that is already generating reports its
gaps between tokens, short requests arriving every few seconds report their
time to first token, and optionally a second long prompt arrives mid-way.

    python3 prefill_block.py http://HEAD:8002 [--words 90000] [--second-long]

Random-word prompts, so nothing is served from the prefix cache.
"""
import argparse
import json
import random
import threading
import time
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu river stone cloud ember maple "
         "harbor lantern meadow orbit prism quartz saddle timber velvet willow").split()


def prompt(n_words, seed):
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n_words)) + "\nReply with one short sentence."


def stream(url, model, text, max_tokens, t0, out):
    """Streams one completion; records (seconds since t0) of the first and every later token."""
    body = {"model": model, "prompt": text, "max_tokens": max_tokens, "temperature": 0, "stream": True}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    out["sent"] = time.time() - t0
    out["tokens"] = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                if json.loads(line[5:])["choices"][0].get("text"):
                    out["tokens"].append(time.time() - t0)
    out["done"] = time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--model", default="glm53")
    ap.add_argument("--words", type=int, default=90000, help="long prompt size (~1.4 tokens per word)")
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--probe-every", type=float, default=2.0)
    ap.add_argument("--second-long", action="store_true")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    t0 = time.time()
    runs, threads = {}, []

    def start(name, text, max_tokens, delay):
        def go():
            time.sleep(delay)
            runs[name] = {}
            stream(url, a.model, text, max_tokens, t0, runs[name])
        th = threading.Thread(target=go)
        th.start()
        threads.append(th)

    start("generating", prompt(40, 1), 3000, 0)          # already streaming when the long prompt lands
    start("long", prompt(a.words, 2), 8, 3)
    if a.second_long:
        start("long2", prompt(a.words // 2, 3), 8, 9)
    for i in range(a.probes):
        start(f"probe{i}", prompt(20, 10 + i), 8, 5 + i * a.probe_every)
    for th in threads:
        th.join()

    long_ = runs["long"]
    lf = long_["tokens"][0] if long_["tokens"] else long_["done"]
    print(f"long prompt: sent {long_['sent']:.1f}s, first token {lf:.1f}s -> prefill {lf - long_['sent']:.1f}s")
    if "long2" in runs:
        l2 = runs["long2"]
        f2 = l2["tokens"][0] if l2["tokens"] else l2["done"]
        print(f"second long: sent {l2['sent']:.1f}s, first token {f2:.1f}s -> waited+prefilled {f2 - l2['sent']:.1f}s")
    toks = runs["generating"]["tokens"]
    gaps = [(b - a_, a_) for a_, b in zip(toks, toks[1:])]
    during = [g for g, at in gaps if long_["sent"] <= at <= lf + 1]
    print(f"generating stream: {len(toks)} tokens; gaps during the long prefill: max {max(during, default=0):.2f}s,"
          f" {sum(1 for g in during if g > 1)} over 1s, {sum(during):.1f}s total in {len(during)} gaps"
          f" (outside it: median {sorted(g for g, _ in gaps)[len(gaps) // 2]:.3f}s)")
    for i in range(a.probes):
        p = runs[f"probe{i}"]
        ttft = (p["tokens"][0] if p["tokens"] else p["done"]) - p["sent"]
        where = "during" if p["sent"] < lf else "after"
        print(f"  probe {i}: sent {p['sent']:5.1f}s ({where} the long prefill), time to first token {ttft:5.2f}s")


if __name__ == "__main__":
    main()
