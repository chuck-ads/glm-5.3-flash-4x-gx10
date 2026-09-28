"""Per-task HumanEval results for several saved runs (inside a --network none container)."""
import glob, os, subprocess, sys
runs = sys.argv[1:]
res = {}
for r in runs:
    for f in sorted(glob.glob(f"/q/{r}/he/*.py")):
        t = os.path.basename(f)[:-3]
        try:
            ok = subprocess.run([sys.executable, f], capture_output=True, timeout=15).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        res.setdefault(t, {})[r] = ok
print("runs:", " ".join(f"{r}={sum(v.get(r, False) for v in res.values())}" for r in runs))
for t, v in sorted(res.items()):
    if len(set(v.values())) > 1 or not any(v.values()):
        print(f"{t:18} " + " ".join("pass" if v.get(r) else "FAIL" for r in runs) + ("   (fails everywhere)" if not any(v.values()) else ""))
