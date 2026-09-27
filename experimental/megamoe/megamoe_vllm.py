# SPDX-License-Identifier: Apache-2.0
"""megamoe for decode-sized batches on vLLM's NVFP4 MoE layers (GB10).

megamoe_rm.cu reads the FlashInfer CUTLASS backend's own processed tensors, so
nothing is copied: batches of at most VLLM_MEGAMOE_MAX_TOKENS tokens take
megamoe and larger ones (prefill) keep CUTLASS. megamoe keeps activations in
16-bit (W4A16, exact in the weights), where CUTLASS quantizes them to FP4.

install() swaps each ModelOptNvFp4FusedMoE method's class for a subclass whose
apply() dispatches by batch size, so every isinstance check still holds. Only
layers whose finalize is synchronous qualify: there the method returns the
weighted top-k sum and the runner owns shared experts and the TP all-reduce,
which is what megamoe produces.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SRC = os.environ.get("VLLM_MEGAMOE_SRC", "/opt/megamoe/megamoe_rm.cu")
_ext = None
_scratch: dict = {}


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        _ext = load("megamoe_rm", [_SRC], extra_cuda_cflags=["-O3", "-gencode=arch=compute_121a,code=sm_121a"])
    return _ext


def _buffers(M: int, topk: int, H: int, I: int, device) -> dict:
    """Scratch shared by every layer (they run one after another), per batch size."""
    key = (M, topk, H, I)
    if key not in _scratch:
        pairs = M * topk
        _scratch[key] = dict(
            xh=torch.empty(M, H, dtype=torch.half, device=device),
            ints=torch.empty(1 + pairs + 8 * pairs, dtype=torch.int32, device=device),
            hbuf=torch.empty(pairs, 8, I, dtype=torch.half, device=device),
            ypart=torch.empty(pairs, H, dtype=torch.float32, device=device))
    return _scratch[key]


def _apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
    M = x.shape[0]
    if M == 0 or M > self._megamoe_max_tokens or x.dtype != torch.bfloat16:
        return super(type(self), self).apply(layer, x, topk_weights, topk_ids, shared_experts,
                                             shared_experts_input)
    w13, w2 = layer.w13_weight, layer.w2_weight
    H, I, topk = w2.shape[1], w13.shape[1] // 2, topk_ids.shape[1]
    s = _buffers(M, topk, H, I, x.device)
    out = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    _ext.forward(x.contiguous(), topk_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous(),
                 w13, layer.w13_weight_scale, layer._megamoe_alpha1, w2, layer.w2_weight_scale,
                 layer._megamoe_alpha2, s["xh"], s["ints"], s["hbuf"], s["ypart"], out,
                 layer._megamoe_limit, self._megamoe_variant)
    return out


def install(model: torch.nn.Module) -> None:
    """Route decode-sized batches of every eligible NVFP4 MoE layer to megamoe."""
    if os.environ.get("VLLM_MEGAMOE") != "1":
        return
    max_tokens = int(os.environ.get("VLLM_MEGAMOE_MAX_TOKENS", "8"))
    variant = int(os.environ.get("VLLM_MEGAMOE_VARIANT", "20"))
    classes: dict[type, type] = {}
    count = 0
    for name, m in model.named_modules():
        method = getattr(m, "quant_method", None)
        if type(method).__name__ != "ModelOptNvFp4FusedMoE" or not hasattr(m, "w13_weight"):
            continue
        kernel = getattr(method, "moe_kernel", None)
        impl = getattr(kernel, "impl", kernel)
        pf = getattr(impl, "prepare_finalize", None)
        if method.is_monolithic or pf is None or pf.supports_async() or m.expert_map is not None:
            logger.warning("megamoe: %s does not qualify; it keeps CUTLASS", name)
            continue
        clamp = getattr(getattr(impl, "fused_experts", None), "gemm1_clamp_limit", None)
        limit = float(clamp.max()) if isinstance(clamp, torch.Tensor) else float("inf")
        m._megamoe_limit = limit
        m._megamoe_alpha1 = (m.w13_weight_scale_2 / m.w13_input_scale).float().contiguous()
        m._megamoe_alpha2 = (m.w2_weight_scale_2 / m.w2_input_scale).float().contiguous()
        cls = type(method)
        if cls not in classes:
            classes[cls] = type("MegaMoE" + cls.__name__, (cls,), {"apply": _apply})
        method.__class__ = classes[cls]
        method._megamoe_max_tokens = max_tokens
        method._megamoe_variant = variant
        count += 1
    if count:
        _load()
    logger.info("megamoe: %d MoE layers take batches of <= %d tokens (variant %d)", count, max_tokens, variant)
