#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the TP=3 serving dirs on one node (run inside the spark-glm53 image).

target: hardlinks of every checkpoint file + a config.json with padded sizes
        and tp_pad_orig (overlay/tp3pad.py pads the tensors at load time).
drafter: DFlash2 (Qwen3, 32 q / 8 kv heads) padded offline to 36 q / 9 kv so
        each rank holds 12 q / 3 kv heads with the same 4:1 grouping. Zero
        q/k/v rows and zero o_proj cols are no-ops.

usage: make_padded.py SRC_MODEL DST_MODEL SRC_DRAFT DST_DRAFT --heads 66 --kda-heads 66 --moe-i 2304
"""
import argparse
import json
import os
import shutil

p = argparse.ArgumentParser()
p.add_argument("src_model"); p.add_argument("dst_model")
p.add_argument("src_draft"); p.add_argument("dst_draft")
p.add_argument("--heads", type=int, default=66)
p.add_argument("--kda-heads", type=int, default=66)
p.add_argument("--moe-i", type=int, default=2304)
p.add_argument("--draft-heads", type=int, default=36)
p.add_argument("--draft-kv-heads", type=int, default=9)
p.add_argument("--tp", type=int, default=3)
a = p.parse_args()

# ---- target ---------------------------------------------------------------
os.makedirs(a.dst_model, exist_ok=True)
for f in os.listdir(a.src_model):
    s, d = os.path.join(a.src_model, f), os.path.join(a.dst_model, f)
    if f == "config.json" or os.path.isdir(s):
        continue
    if not os.path.exists(d):
        os.link(s, d)
cfg = json.load(open(os.path.join(a.src_model, "config.json")))
tc = cfg["text_config"]
orig = {"num_attention_heads": tc["num_attention_heads"],
        "linear_num_heads": tc["linear_num_heads"],
        "moe_intermediate_size": tc["moe_intermediate_size"]}
tc["tp_pad_orig"] = orig
tc["num_attention_heads"] = a.heads
tc["num_key_value_heads"] = a.heads
tc["linear_num_heads"] = a.kda_heads
tc["linear_attn_config"]["num_heads"] = a.kda_heads
tc["moe_intermediate_size"] = a.moe_i
tmp = os.path.join(a.dst_model, "config.json.tmp")
json.dump(cfg, open(tmp, "w"), indent=2)
os.replace(tmp, os.path.join(a.dst_model, "config.json"))
print("target", a.dst_model, orig, "->", a.heads, a.kda_heads, a.moe_i)

# ---- drafter --------------------------------------------------------------
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

os.makedirs(a.dst_draft, exist_ok=True)
dc = json.load(open(os.path.join(a.src_draft, "config.json")))
H0, KV0, hd = dc["num_attention_heads"], dc["num_key_value_heads"], dc["head_dim"]
H1, KV1 = a.draft_heads, a.draft_kv_heads
assert H0 // KV0 == H1 // KV1 and H1 % a.tp == 0 and KV1 % a.tp == 0
sd = load_file(os.path.join(a.src_draft, "model.safetensors"))


def pad(t, dim, size):
    shape = list(t.shape); shape[dim] = size
    out = t.new_zeros(shape)
    out.narrow(dim, 0, t.shape[dim]).copy_(t)
    return out


n = 0
for k in list(sd):
    t = sd[k]
    if k.endswith("self_attn.q_proj.weight"):
        sd[k] = pad(t, 0, H1 * hd); n += 1
    elif k.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
        sd[k] = pad(t, 0, KV1 * hd); n += 1
    elif k.endswith("self_attn.o_proj.weight"):
        sd[k] = pad(t, 1, H1 * hd); n += 1
save_file(sd, os.path.join(a.dst_draft, "model.safetensors"), metadata={"format": "pt"})
dc["num_attention_heads"], dc["num_key_value_heads"] = H1, KV1
json.dump(dc, open(os.path.join(a.dst_draft, "config.json"), "w"), indent=2)
for f in os.listdir(a.src_draft):
    if f not in ("config.json", "model.safetensors") and os.path.isfile(os.path.join(a.src_draft, f)):
        shutil.copy(os.path.join(a.src_draft, f), a.dst_draft)
print("drafter", a.dst_draft, f"padded {n} tensors: {H0}/{KV0} -> {H1}/{KV1} heads")
