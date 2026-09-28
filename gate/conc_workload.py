"""Aggregate decode tok/s with N concurrent streams of one workload, each stream a
different prompt (identical prompts would route to the same experts).

    python3 conc_workload.py WORKLOAD N [N ...]      WORKLOAD: code | prose | structured | mixed
"""
import json, os, re, statistics, sys, threading, time, urllib.request

BASE = os.environ.get("GATE_URL", "http://127.0.0.1:8002")
MODEL = os.environ.get("GATE_MODEL", "glm53")
PROMPTS = {
    "code": [f"Write {t} in Python. Code only." for t in (
        "a red-black tree with insert, delete and rebalancing", "an LRU cache with O(1) get and put",
        "a trie with insert, search and prefix listing", "a binary min-heap with push, pop and heapify",
        "a JSON tokenizer that yields typed tokens", "a fixed-size thread pool with a work queue",
        "Dijkstra's shortest path over an adjacency list", "a dense matrix class with multiply and transpose")],
    "prose": [f"Explain {t}, in flowing prose. No code, no lists." for t in (
        "how a hash map works", "how TCP opens and closes a connection", "how photosynthesis stores energy",
        "how a tracing garbage collector works", "how compound interest grows savings", "how vaccines train immunity",
        "how plate tectonics shapes continents", "how a compiler turns source into machine code")],
    "structured": [f"{t} No commentary." for t in (
        "Count from 1 to 200, comma separated.", "List the integers from 300 to 480, comma separated.",
        "List the multiples of 3 from 3 to 540, comma separated.", "Count down from 220 to 1, comma separated.",
        "List the even numbers from 2 to 380, comma separated.", "List the squares of 1 to 90, comma separated.",
        "List the integers from 1000 to 1150, comma separated.", "List the multiples of 7 from 7 to 1050, comma separated.")],
}
SUFFIX = ["", " Be thorough."]
# Streams past the first 16 draw from these, so large batches do not repeat
# prompts (repeated greedy prompts route to the same experts).
EXTRA = {
    "code": [f"Write {t} in Python. Code only." for t in (
        "a union-find with path compression", "a bloom filter with k hash functions", "an interval tree with overlap queries",
        "a skip list with insert and search", "a ring buffer with overwrite", "a token bucket rate limiter",
        "a consistent hash ring", "an A* path finder on a grid", "a topological sort with cycle detection",
        "a CSV parser that handles quoted fields", "a Huffman encoder and decoder", "a sudoku solver with backtracking",
        "a B-tree of order 4 with insert", "a Fenwick tree with range sums", "a segment tree with lazy propagation",
        "an event loop with timers", "a regex matcher for . * and ?", "a markdown to HTML converter for headings and lists",
        "a merge sort and a quicksort with benchmarks", "a k-means clustering routine", "a simple stack-based VM",
        "a URL router with path parameters", "an INI file parser", "a priority task scheduler",
        "a diff of two lists by longest common subsequence", "a base64 encoder and decoder", "a sparse matrix in CSR form",
        "a minimal HTTP request parser", "a finite state machine for a vending machine", "a memoizing decorator with TTL")],
    "prose": [f"Explain {t}, in flowing prose. No code, no lists." for t in (
        "how a refrigerator moves heat", "how the heart pumps blood", "how a jet engine makes thrust",
        "how DNS resolves a name", "how a bill becomes law", "how bread rises", "how glaciers carve valleys",
        "how a transistor switches", "how the stock market sets prices", "how bees communicate",
        "how GPS finds a position", "how a lock and key work", "how clouds form rain", "how a piano makes sound",
        "how the moon causes tides", "how antibiotics kill bacteria", "how a camera sensor records light",
        "how inflation erodes money", "how a river delta forms", "how public key encryption works",
        "how muscles contract", "how a wind turbine makes power", "how the immune system spots a virus",
        "how a library catalogs books", "how coffee is roasted", "how an elevator stays safe",
        "how sleep restores the brain", "how a sailboat moves upwind", "how volcanoes erupt", "how a database index speeds queries")],
    "structured": [f"List the multiples of {m} from {m * a} to {m * (a + 150)}, comma separated. No commentary."
                   for m in (2, 3, 4, 5, 6, 7, 8, 9, 11, 13) for a in (10, 400, 900)],
}
STYLE = ["", " Keep it short.", " Be thorough.", " Aim it at a beginner.", " Aim it at an expert.",
         " Use plain words.", " Be precise.", " Be complete."]


def prompt_for(work, i):
    if work == "mixed":
        return prompt_for(("code", "prose", "structured")[i % 3], i // 3)
    if i < 16:
        ps = PROMPTS[work]
        return ps[i % len(ps)] + SUFFIX[(i // len(ps)) % len(SUFFIX)]
    ex = EXTRA[work]
    j = i - 16
    return ex[j % len(ex)] + STYLE[(j // len(ex)) % len(STYLE)]


def metrics():
    raw = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    get = lambda k: sum(float(v) for v in re.findall(rf"^vllm:{k}\S*\s+([0-9.e+]+)$", raw, re.M))
    return get("spec_decode_num_accepted_tokens_total"), get("spec_decode_num_drafts_total")


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


work = sys.argv[1]
for N in [int(a) for a in sys.argv[2:]]:
    prompts = [prompt_for(work, i) for i in range(N)]
    a0, d0 = metrics()
    out = [None] * N
    ts = [threading.Thread(target=one, args=(prompts[i], out, i)) for i in range(N)]
    for t in ts: t.start()
    for t in ts: t.join()
    a1, d1 = metrics()
    t0 = min(o[1] for o in out); t1 = max(o[2] for o in out)
    per = [o[0] / (o[2] - o[1]) for o in out]
    al = 1 + (a1 - a0) / max(d1 - d0, 1)  # tokens per verify step per request
    print(f"{work:<10} streams {N:2d}: aggregate {sum(o[0] for o in out) / (t1 - t0):6.1f} tok/s,"
          f" per stream median {statistics.median(per):5.1f}, accepted per step {al:.2f}", flush=True)
