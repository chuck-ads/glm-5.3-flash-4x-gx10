#!/usr/bin/env python3
"""experimental/fixes/gb10_sparse_mla.py at head counts that are not powers of two.

TP=3 leaves 22 MLA heads per rank (66 padded / 3). The kernel used to take
tl.arange(0, H), which needs a power of two; it now tiles at the next power of
two (>= 16) with a head mask. This checks both launch paths (one program per
token, and the split + combine path decode takes) against a float32 torch
reference, at H = 16 and 32 (unchanged tiles) and 22 and 24 (masked), and
H = 11 (TP=6: 66 / 6, a 16-row tile with 5 rows masked).

Pass: every case within 5e-3 relative error of the reference, and the masked
cases no worse than twice the H=16 error. On one GB10:

    mkdir -p /tmp/smla && cp dev/patch-tests/_glm53_tp3_sparse_mla_test.py \\
        experimental/fixes/gb10_sparse_mla.py /tmp/smla/
    sudo docker run --rm --gpus all -v /tmp/smla:/t -w /t \\
        --entrypoint python3 <glm53 image> /t/_glm53_tp3_sparse_mla_test.py
"""
import importlib.util
import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
path = next(p for p in (os.path.join(here, "gb10_sparse_mla.py"),
                        os.path.join(here, "../../experimental/fixes/gb10_sparse_mla.py")) if os.path.exists(p))
spec = importlib.util.spec_from_file_location("gb10_sparse_mla", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

dev = torch.device("cuda")
D, S, TOPK = 512, 8192, 512


def reference(q, kv, idx, lens, sm_scale, kv_scale):
    kvf = kv.float() * kv_scale
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for t in range(q.shape[0]):
        rows = kvf[idx[t, :lens[t]].long()]                      # [n, D]
        s = (q[t].float() @ rows.T) * sm_scale                    # [H, n]
        out[t] = torch.softmax(s, -1) @ rows
    return out


def case(H, T, splits, g):
    q = torch.randn(T, H, D, generator=g, device=dev).to(torch.bfloat16)
    kv = (torch.randn(S, D, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    lens = torch.randint(TOPK // 2, TOPK + 1, (T,), generator=g, device=dev).int()
    idx = torch.stack([torch.randperm(S, generator=g, device=dev)[:TOPK] for _ in range(T)]).int()
    for t in range(T):
        idx[t, lens[t]:] = -1
    sm_scale, kv_scale = D ** -0.5, 0.75
    got = m.sparse_mla(q, kv, idx, lens, sm_scale, kv_scale, splits=splits).float()
    ref = reference(q, kv, idx, lens, sm_scale, kv_scale)
    return ((got - ref).norm() / ref.norm()).item()


g = torch.Generator(device=dev).manual_seed(0)
errs, fails = {}, 0
for T, splits in ((64, 1), (4, 8)):
    for H in (11, 16, 22, 24, 32):
        e = case(H, T, splits, g)
        errs[(H, T)] = e
        print(f"H={H:2d} T={T:2d} splits={splits}: rel err {e:.2e}", flush=True)
for (H, T), e in errs.items():
    if e > 5e-3 or e > 2 * max(errs[(16, T)], 1e-4):
        print(f"FAIL H={H} T={T}: {e:.2e}"); fails += 1
print("PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
