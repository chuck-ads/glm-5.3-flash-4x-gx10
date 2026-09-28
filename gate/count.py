"""Greedy count to 200, N times: any wrong line is a corrupt run. usage: count.py [N]"""
import json, os, sys, urllib.request
BASE = os.environ.get("GATE_URL", "http://127.0.0.1:8002")
MODEL = os.environ.get("GATE_MODEL", "glm53")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
bad_runs = 0
for i in range(N):
    body = {"model": MODEL, "max_tokens": 700, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Count from 1 to 200, one number per line, digits only."}]}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=300)); txt = d["choices"][0]["message"]["content"] or ""
    nums = [l.strip() for l in txt.strip().split("\n")]
    clean = nums == [str(k) for k in range(1, 201)]
    bad_runs += (not clean)
    first_bad = next((j for j, (a, b) in enumerate(zip(nums, [str(k) for k in range(1, 201)])) if a != b), None)
    print(f"run {i}: {'CLEAN' if clean else 'CORRUPT'}  tokens={d['usage']['completion_tokens']}  lines={len(nums)}  first_bad_line={first_bad}")
print(f"corrupt {bad_runs}/{N}")
