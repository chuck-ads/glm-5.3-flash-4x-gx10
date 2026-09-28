"""GSM8K (first N) and HumanEval generations against a running server; HumanEval
programs are written out for a separate no-network run. usage: quality.py LABEL [gsm_n]"""
import concurrent.futures as cf, json, os, re, sys, urllib.request

LABEL = sys.argv[1]; GSM_N = int(sys.argv[2]) if len(sys.argv) > 2 else 250
BASE = "http://127.0.0.1:8002/v1/chat/completions"
OUT = f"/home/admin/quality/{LABEL}"; os.makedirs(f"{OUT}/he", exist_ok=True)

def chat(prompt, max_tokens=1024):
    body = {"model": "glm53", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "seed": 1234, "chat_template_kwargs": {"thinking": False}}
    r = urllib.request.Request(BASE, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())["choices"][0]["message"]["content"] or ""

def num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try: return float(s)
    except ValueError: return None

gsm = [json.loads(l) for l in open("/home/admin/quality/gsm8k_test.jsonl")][:GSM_N]
def do_gsm(q):
    txt = chat("Solve the problem. Show brief working, then give the final answer on the last line as '#### <number>'.\n\n" + q["question"])
    m = re.findall(r"####\s*([-\d.,$]+)", txt) or re.findall(r"(-?[\d,]*\.?\d+)", txt)
    got = num(m[-1]) if m else None
    gold = num(q["answer"].split("####")[-1])
    return got is not None and gold is not None and abs(got - gold) < 1e-6, txt
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(do_gsm, gsm))
acc = sum(ok for ok, _ in res) / len(res)
json.dump([t for _, t in res], open(f"{OUT}/gsm8k.json", "w"))
print(f"{LABEL} GSM8K first {len(gsm)}: {acc * 100:.1f}%", flush=True)

he = [json.loads(l) for l in open("/home/admin/quality/HumanEval.jsonl")]
PRELUDE = "from typing import *\nimport math, re, string, collections, itertools, functools, heapq, bisect\n"
def do_he(p):
    txt = chat("Complete the following Python function. Reply with the whole function in one ```python code block and nothing else.\n\n" + p["prompt"])
    m = re.findall(r"```(?:python)?\n(.*?)```", txt, re.S)
    code = m[0] if m else txt
    if f"def {p['entry_point']}" not in code:
        code = p["prompt"] + code
    prog = PRELUDE + code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
    open(f"{OUT}/he/{p['task_id'].replace('/', '_')}.py", "w").write(prog)
with cf.ThreadPoolExecutor(8) as ex:
    list(ex.map(do_he, he))
print(f"{LABEL} HumanEval: {len(he)} programs written", flush=True)
