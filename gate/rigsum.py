"""RigMark JSON summary: per-workload gate and median decode, cold prefill per depth."""
import json, sys
d = json.load(open(sys.argv[1]))
for w, v in d["decode"].items():
    g = v.get("completion_gate", {})
    print(f"{w:12s} {v['decode_tokens_per_second']['median']:7.1f} tok/s  gate {g.get('passed')}/{g.get('total')}")
for depth, v in d["prefill"].items():
    print(f"prefill {depth}: cold {v['cold']['effective_prefill_tokens_per_second']['median']:,.0f} tok/s")
