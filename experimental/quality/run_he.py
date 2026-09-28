"""Run HumanEval programs (inside a --network none container): pass@1."""
import glob, subprocess, sys
files = sorted(glob.glob(f"{sys.argv[1]}/he/*.py"))
ok = 0
for f in files:
    try:
        ok += subprocess.run([sys.executable, f], capture_output=True, timeout=15).returncode == 0
    except subprocess.TimeoutExpired:
        pass
print(f"HumanEval pass@1: {ok}/{len(files)} = {ok / len(files) * 100:.1f}%")
