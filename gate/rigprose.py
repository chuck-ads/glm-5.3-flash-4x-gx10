"""RigMark's prose decode, alone: the identical requests (comparison id, seed, nonces,
reasoning_effort=low) and its decode-rate formula, plus DFlash acceptance from /metrics.
usage: rigprose.py BASE_URL [runs] [workload]"""
import json, os, re, sys, time, urllib.request
sys.path.insert(0, os.environ.get("RIGMARK_DIR", "rigmark"))
import bench as rb
from pathlib import Path

BASE = sys.argv[1]; RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
WL = sys.argv[3] if len(sys.argv) > 3 else "prose"
prompts, _ = rb.load_prompts(Path(rb.__file__).parent / "prompts.json")
client = rb.Client(BASE, "", 600)

def metrics():
    raw = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    get = lambda k: sum(float(v) for v in re.findall(rf"^vllm:{k}\S*\s+([0-9.e+]+)$", raw, re.M))
    return (get("spec_decode_num_accepted_tokens_total"), get("spec_decode_num_draft_tokens_total"),
            get("spec_decode_num_drafts_total"))

rates = []; dsec = 0.0
a0, d0, s0 = metrics()
for i in range(RUNS):
    nonce = rb.nonce("ringside-redhat-rowsplit-20260926", "decode", WL, i + 1)
    payload = rb.build_chat_payload(os.environ.get("GATE_MODEL", "glm53"), prompts["system"], f"Request nonce: {nonce}\n\n{prompts['workloads'][WL]}",
                                    4096, 20260905, {"chat_template_kwargs": {"reasoning_effort": "low"}})
    row = client.stream("/v1/chat/completions", payload)
    rates.append(row["decode_tokens_per_second"]); dsec += row["decode_seconds"]
    print(f"  run {i + 1}: {row['decode_tokens_per_second']:.1f} tok/s, {row['completion_tokens']} tokens, TTFT {row['ttft_seconds']:.3f}s", flush=True)
a1, d1, s1 = metrics()
acc = (a1 - a0) / (d1 - d0) if d1 > d0 else float("nan")
steps = s1 - s0
print(f"{WL}: {sum(rates) / len(rates):.1f} tok/s mean, acceptance {acc * 100:.1f}%, "
      f"{dsec * 1000 / steps if steps else float('nan'):.2f} ms/step over {int(steps)} steps")
