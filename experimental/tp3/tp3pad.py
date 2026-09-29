# SPDX-License-Identifier: Apache-2.0
"""Zero-pad GLM-5.3-Flash weights at load time so uneven dims split at TP=3.

The served config.json carries padded sizes (num_attention_heads,
linear_num_heads, moe_intermediate_size) plus ``tp_pad_orig`` with the
checkpoint's real ones. Every checkpoint tensor that is sized by one of those
dims is zero-extended here, before vLLM's loaders narrow it per rank, so the
extra heads / intermediate columns land at the end (the last rank's slice).

Zero rows/cols are exact no-ops:
- MLA: a zero head has q=0 and zero kv_b rows (so V=0) and zero o_proj cols.
- KDA: q=k=v=0 keeps the recurrent state at 0, the gated RMSNorm of a zero
  vector is 0, and the o_proj cols are 0.
- MoE / shared expert: gate=up=0 gives silu(0)*0=0, down cols are 0.
NVFP4 tensors: packed uint8 zero = two FP4 +0 codes, fp8 block scale 0.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _pad(t: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    if t.shape[dim] == size:
        return t
    assert t.shape[dim] < size, (tuple(t.shape), dim, size)
    src = t.view(torch.uint8) if t.element_size() == 1 else t
    shape = list(src.shape)
    shape[dim] = size
    out = src.new_zeros(shape)
    out.narrow(dim, 0, src.shape[dim]).copy_(src)
    return out.view(t.dtype) if t.element_size() == 1 else out


def _cols(name: str, t: torch.Tensor, logical: int) -> int:
    """Physical width of a padded input dim for this tensor's format."""
    if name.endswith(".weight_scale"):
        return logical // 16
    if t.dtype == torch.uint8:
        return logical // 2
    return logical


def pad_weights(weights, config):
    orig = getattr(config, "tp_pad_orig", None)
    if not orig:
        yield from weights
        return
    H0, H1 = orig["num_attention_heads"], config.num_attention_heads
    K0, K1 = orig["linear_num_heads"], config.linear_num_heads
    I0, I1 = orig["moe_intermediate_size"], config.moe_intermediate_size
    qk = config.qk_nope_head_dim + config.qk_rope_head_dim
    kvb = config.qk_nope_head_dim + config.v_head_dim
    v = config.v_head_dim
    hd = config.linear_head_dim
    ns = config.n_shared_experts or 1
    n = 0
    for name, t in weights:
        new = t
        if ".self_attn." in name and ".indexer." not in name:
            s = name.rsplit(".self_attn.", 1)[1]
            if s == "q_b_proj.weight" and t.shape[0] == H0 * qk:
                new = _pad(t, 0, H1 * qk)
            elif s == "kv_b_proj.weight" and t.shape[0] == H0 * kvb:
                new = _pad(t, 0, H1 * kvb)
            elif s == "o_proj.weight" and t.shape[1] == H0 * v:
                new = _pad(t, 1, H1 * v)
            elif s == "o_proj.weight" and t.shape[1] == K0 * hd:
                new = _pad(t, 1, K1 * hd)
            elif s in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                       "f_b_proj.weight", "g_b_proj.weight", "dt_bias",
                       "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight") \
                    and t.shape[0] == K0 * hd:
                new = _pad(t, 0, K1 * hd)
            elif s in ("b_proj.weight", "A_log") and t.shape[0] == K0:
                new = _pad(t, 0, K1)
        elif ".mlp.experts." in name or ".mlp.shared_experts." in name:
            i0, i1 = (I0 * ns, I1 * ns) if ".shared_experts." in name else (I0, I1)
            if name.endswith((".weight", ".weight_scale")):
                if (".gate_proj." in name or ".up_proj." in name) and t.shape[0] == i0:
                    new = _pad(t, 0, i1)
                elif ".down_proj." in name and t.shape[1] == _cols(name, t, i0):
                    new = _pad(t, 1, _cols(name, t, i1))
        if new is not t:
            n += 1
        yield name, new
    logger.info("tp3pad: zero-padded %d tensors (heads %d->%d, kda %d->%d, moe I %d->%d)",
                n, H0, H1, K0, K1, I0, I1)
