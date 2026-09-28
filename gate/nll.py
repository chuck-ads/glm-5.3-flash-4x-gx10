# Mean NLL per token of fixed texts under the served model (prompt logprobs).
# usage: nll.py record OUT.json | nll.py compare [REF.json]
# nll-ref.json was recorded with the dense linears in bf16 (fp8.yaml left out).
import json, math, os, sys, urllib.request
BASE = os.environ.get("GATE_URL", "http://127.0.0.1:8002")
MODEL = os.environ.get("GATE_MODEL", "glm53")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "texts") + "/"
TEXTS = {"notes": open(D + "notes.md").read()[:14000],
         "arx.cu": open(D + "arx.cu").read()[:14000], "snap.py": open(D + "snap.py").read()[:14000]}
def nll(text):
    body = json.dumps({"model": MODEL, "prompt": text, "max_tokens": 1, "temperature": 0,
                       "prompt_logprobs": 0, "cache_salt": os.urandom(8).hex()}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=600).read())
    lps = []
    for entry in r["choices"][0]["prompt_logprobs"][1:]:
        if entry:
            lps.append(max(v["logprob"] for v in entry.values()) if len(entry) == 1 else
                       next(v["logprob"] for v in entry.values() if v.get("rank") is not None and v["rank"] >= 1))
    return lps
res = {k: nll(t) for k, t in TEXTS.items()}
if sys.argv[1] == "record":
    json.dump(res, open(sys.argv[2], "w"))
    for k, v in res.items(): print(f"{k:10} tokens {len(v):5d}  mean NLL {-sum(v)/len(v):.4f}")
else:
    ref = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "nll-ref.json")))
    for k, v in res.items():
        r = ref[k]; n = min(len(r), len(v))
        d = [abs(a - b) for a, b in zip(v[:n], r[:n])]
        print(f"{k:10} tokens {n:5d}  mean NLL {-sum(v)/len(v):.4f} (ref {-sum(r)/len(r):.4f}, "
              f"delta {(-sum(v)/len(v)) - (-sum(r)/len(r)):+.4f})  mean |dlogprob| {sum(d)/n:.4f}  max {max(d):.3f}")
