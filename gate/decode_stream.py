#!/usr/bin/env python3
"""Per-workload decode for glm53, measured the way the winning-config numbers were:
thinking off, 512 output tokens, temp 0, streaming, drafter acceptance from /metrics.
Median of 3. Usage: decode_stream.py
"""
import json, os, statistics, sys, time, urllib.request
BASE = os.environ.get("GATE_URL", "http://127.0.0.1:8002")
MODEL = os.environ.get("GATE_MODEL", "glm53")
def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    g = lambda k: sum(float(l.rsplit(" ", 1)[1]) for l in t.splitlines() if l.startswith(k + "{") or l.startswith(k + " "))
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total"), g("vllm:spec_decode_num_draft_tokens_total")
def stream(prompt, n):
    body = {"model": MODEL, "max_tokens": n, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); first = None; toks = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if first is None and d.get("choices") and (d["choices"][0]["delta"].get("content") or d["choices"][0]["delta"].get("reasoning")):
                first = time.monotonic()
            if d.get("usage"): toks = d["usage"]["completion_tokens"]
    end = time.monotonic()
    return toks, toks / (end - first) if first and end > first else 0.0
W = {"structured": "Count from 1 to 200, one number per line, digits only.",
     "code": "Write a complete Python implementation of a red-black tree with insert, delete and in-order traversal.",
     "prose": "Explain in detail how a hash map handles collisions, with examples."}
stream("hi", 4)
for name, p in W.items():
    rates, accs = [], []
    for _ in range(3):
        d0, a0, t0 = metrics(); n, r = stream(p, 512); d1, a1, t1 = metrics()
        rates.append(r); accs.append((a1 - a0) / (t1 - t0) if t1 > t0 else float("nan"))
    print(f"{name:10} decode {statistics.median(rates):6.1f} tok/s  acceptance {100*statistics.median(accs):5.1f}%  ({n} tok)", flush=True)
