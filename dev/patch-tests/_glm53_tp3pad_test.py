#!/usr/bin/env python3
"""experimental/tp3/tp3pad.py: the load-time zero padding behind TP=3.

Builds a small GLM-5.3-Flash-shaped set of tensors (one MLA layer, one KDA
layer, routed experts in bf16 and NVFP4 layout, a shared expert) and runs
pad_weights with a config that pads heads 4->6, KDA heads 4->6 and the MoE
intermediate 32->48. Checks:
  - every sized tensor reaches the padded shape, the original block is
    byte-identical and the padding is zero (uint8 NVFP4 and scales included)
  - tensors that are not sized by a padded dim pass through untouched
  - a bf16 expert forward, silu(x Wg) * (x Wu) Wd, is bit-identical padded
  - with no tp_pad_orig in the config, weights pass through as-is
CPU only, no vLLM needed (vllm.logger is stubbed when absent):

    python3 dev/patch-tests/_glm53_tp3pad_test.py
"""
import importlib.util
import logging
import os
import sys
import types

import torch

try:
    import vllm.logger  # noqa: F401
except Exception:
    stub = types.ModuleType("vllm.logger")
    stub.init_logger = logging.getLogger
    sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    sys.modules["vllm.logger"] = stub

here = os.path.dirname(os.path.abspath(__file__))
path = next(p for p in (os.path.join(here, "tp3pad.py"),
                        os.path.join(here, "../../experimental/tp3/tp3pad.py")) if os.path.exists(p))
spec = importlib.util.spec_from_file_location("tp3pad", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

H0, H1, K0, K1, I0, I1 = 4, 6, 4, 6, 32, 48
nope, rope, vd, hd, hid = 8, 4, 8, 16, 64
cfg = types.SimpleNamespace(num_attention_heads=H1, linear_num_heads=K1, moe_intermediate_size=I1,
                            qk_nope_head_dim=nope, qk_rope_head_dim=rope, v_head_dim=vd,
                            linear_head_dim=hd, n_shared_experts=1,
                            tp_pad_orig={"num_attention_heads": H0, "linear_num_heads": K0,
                                         "moe_intermediate_size": I0})
g = torch.Generator().manual_seed(0)
r = lambda *s: torch.randn(*s, generator=g).to(torch.bfloat16)
u8 = lambda *s: torch.randint(1, 255, s, generator=g, dtype=torch.uint8)
L = "model.layers.0"
W = {  # name: (tensor, padded shape or None if untouched)
    f"{L}.self_attn.q_b_proj.weight": (r(H0 * (nope + rope), 32), (H1 * (nope + rope), 32)),
    f"{L}.self_attn.kv_b_proj.weight": (r(H0 * (nope + vd), 16), (H1 * (nope + vd), 16)),
    f"{L}.self_attn.o_proj.weight": (r(hid, H0 * vd), (hid, H1 * vd)),
    f"{L}.self_attn.kv_a_proj_with_mqa.weight": (r(16 + rope, hid), None),
    f"{L}.self_attn.indexer.wq_b.weight": (r(H0 * (nope + rope), 32), None),
    "model.layers.1.self_attn.q_proj.weight": (r(K0 * hd, hid), (K1 * hd, hid)),
    "model.layers.1.self_attn.q_conv1d.weight": (r(K0 * hd, 1, 4), (K1 * hd, 1, 4)),
    "model.layers.1.self_attn.b_proj.weight": (r(K0, hid), (K1, hid)),
    "model.layers.1.self_attn.A_log": (r(K0), (K1,)),
    "model.layers.1.self_attn.dt_bias": (r(K0 * hd), (K1 * hd,)),
    "model.layers.1.self_attn.o_proj.weight": (r(hid, K0 * hd), (hid, K1 * hd)),
    f"{L}.mlp.experts.0.gate_proj.weight": (r(I0, hid), (I1, hid)),
    f"{L}.mlp.experts.0.up_proj.weight": (r(I0, hid), (I1, hid)),
    f"{L}.mlp.experts.0.down_proj.weight": (r(hid, I0), (hid, I1)),
    f"{L}.mlp.experts.1.gate_proj.weight": (u8(I0, hid // 2), (I1, hid // 2)),
    f"{L}.mlp.experts.1.gate_proj.weight_scale": (u8(I0, hid // 16), (I1, hid // 16)),
    f"{L}.mlp.experts.1.down_proj.weight": (u8(hid, I0 // 2), (hid, I1 // 2)),
    f"{L}.mlp.experts.1.down_proj.weight_scale": (u8(hid, I0 // 16), (hid, I1 // 16)),
    f"{L}.mlp.experts.1.down_proj.weight_scale_2": (r(1), None),
    f"{L}.mlp.shared_experts.gate_proj.weight": (r(I0, hid), (I1, hid)),
    f"{L}.mlp.shared_experts.down_proj.weight": (r(hid, I0), (hid, I1)),
    f"{L}.mlp.gate.weight": (r(8, hid), None),
}
fails = 0
out = dict(m.pad_weights(((k, v[0]) for k, v in W.items()), cfg))
for name, (t, shape) in W.items():
    p = out[name]
    if shape is None:
        ok = p is t
    else:
        a, b = (x.view(torch.uint8) if x.element_size() == 1 else x for x in (t, p))
        sl = tuple(slice(0, n) for n in t.shape)
        pad = b.clone(); pad[sl] = 0
        ok = tuple(p.shape) == shape and p.dtype == t.dtype and torch.equal(b[sl], a) and not pad.any()
    if not ok:
        print(f"FAIL {name}: {tuple(t.shape)} -> {tuple(p.shape)}"); fails += 1

x = r(3, hid).float()
e = lambda gw, uw, dw: (torch.nn.functional.silu(x @ gw.float().T) * (x @ uw.float().T)) @ dw.float().T
E = f"{L}.mlp.experts.0."
before = e(W[E + "gate_proj.weight"][0], W[E + "up_proj.weight"][0], W[E + "down_proj.weight"][0])
after = e(out[E + "gate_proj.weight"], out[E + "up_proj.weight"], out[E + "down_proj.weight"])
if not torch.equal(before, after):
    print("FAIL expert forward differs after padding"); fails += 1

plain = types.SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k != "tp_pad_orig"})
if any(a is not b for (_, a), b in zip(m.pad_weights(((k, v[0]) for k, v in W.items()), plain),
                                        (v[0] for v in W.values()))):
    print("FAIL without tp_pad_orig the weights were changed"); fails += 1

print(f"{len(W)} tensors checked, expert forward, pass-through: " + ("PASS" if not fails else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
