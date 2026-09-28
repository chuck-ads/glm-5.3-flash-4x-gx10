# Aggregate and per-stream decode tok/s with N concurrent streams (streamed,
# timed from each stream's first token). Prompts rotate across workloads.
import json, os, statistics, sys, threading, time, urllib.request
BASE = os.environ.get("GATE_URL", "http://127.0.0.1:8002")
MODEL = os.environ.get("GATE_MODEL", "glm53")
PROMPTS = ["Write a red-black tree in Python with insert, delete and rebalancing. Code only.",
           "Explain how a hash map works, in flowing prose. No code, no lists.",
           "Write a complete Python implementation of a red-black tree with insert, delete and in-order traversal.",
           "Explain in detail how a hash map handles collisions, with examples."]


def one(prompt, out, i):
    body = json.dumps({"model": MODEL, "max_tokens": 512, "temperature": 0, "stream": True,
                       "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=900)
    first = n = None
    for line in r:
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        if first is None and d.get("choices"):
            first = time.monotonic()
        if d.get("usage"):
            n = d["usage"]["completion_tokens"]
    out[i] = (n, first, time.monotonic())


for N in [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8]:
    out = [None] * N
    ts = [threading.Thread(target=one, args=(PROMPTS[i % 4], out, i)) for i in range(N)]
    for t in ts: t.start()
    for t in ts: t.join()
    t0 = min(o[1] for o in out); t1 = max(o[2] for o in out)
    per = [o[0] / (o[2] - o[1]) for o in out]
    print(f"streams {N:2d}: aggregate {sum(o[0] for o in out) / (t1 - t0):6.1f} tok/s, per stream median {statistics.median(per):6.1f} (min {min(per):.1f})", flush=True)
