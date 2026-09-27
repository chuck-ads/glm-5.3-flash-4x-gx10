# SPDX-License-Identifier: Apache-2.0
"""FP8 weights for the dense linears a checkpoint left in bf16.

GLM-5.3-Flash's NVFP4 checkpoint quantizes only the routed experts. About 4 GB
per rank of dense projections (the KDA input projection, o_proj, the shared
experts, the MLA and indexer query projections) stay bf16, and decode reads
all of it every step. On GB10 those GEMMs run at the memory roofline, so the
bytes are the cost: FP8 halves them.

Weights get one scale per output channel, activations one per token, and the
GEMM is CUTLASS's scaled_mm, which reaches ~90% of the FP8 read roofline from
M = 1 upwards. The router gate stays bf16 (routing is sensitive), and so do
layers whose weight other code reads directly (kv_b_proj, the indexer's
wk_weights_proj). The lm_head converts too unless VLLM_DENSE_FP8_LM_HEAD=0.
"""

import os
import re

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger(__name__)

_EXCLUDE = re.compile(r"(^|\.)(gate|kv_b_proj|wk_weights_proj|index_kpool_compress_gate)$|(^|\.)visual\.")


def _fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                bias: torch.Tensor | None) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.shape[0] == 0:
        return x.new_empty(*shape[:-1], weight.shape[0])
    xq, xs = ops.scaled_fp8_quant(x2.contiguous(), use_per_token_if_dynamic=True)
    out = ops.cutlass_scaled_mm(xq, weight.t(), xs, scale, x.dtype, bias)
    return out.reshape(*shape[:-1], weight.shape[0])


class Fp8DenseLinearMethod(LinearMethodBase):
    """A converted layer's forward: per-token FP8 activations x FP8 weight.

    Not an UnquantizedLinearMethod: code that sees one assumes a bf16 weight it
    can use directly (the DFlash drafter concatenates its KV weights)."""

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError("layers are converted after loading")

    def apply(self, layer, x, bias=None):
        return _fp8_linear(x, layer.weight, layer.weight_scale, bias)


class Fp8LMHeadMethod:
    """The lm_head's logits (the only path a ParallelLMHead takes at decode)."""

    def __init__(self, inner):
        self.inner = inner

    def apply(self, layer, x, bias=None):
        return _fp8_linear(x, layer.weight, layer.weight_scale, bias)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _quantize(module: torch.nn.Module) -> int:
    w = module.weight.data
    wq, ws = ops.scaled_fp8_quant(w.contiguous(), use_per_token_if_dynamic=True)
    module.weight = Parameter(wq, requires_grad=False)
    module.weight_scale = Parameter(ws.view(1, -1).to(torch.float32).contiguous(), requires_grad=False)
    return w.numel() * w.element_size() - wq.numel()


@torch.no_grad()
def convert(model: torch.nn.Module) -> None:
    """Convert every eligible bf16 linear in place. Idempotent."""
    if os.environ.get("VLLM_DENSE_FP8") != "1":
        return
    lm_head = os.environ.get("VLLM_DENSE_FP8_LM_HEAD", "1") == "1"
    saved, count = 0, 0
    for name, m in model.named_modules():
        w = getattr(m, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.dim() != 2:
            continue
        if w.shape[0] % 16 or w.shape[1] % 16 or _EXCLUDE.search(name):
            continue
        method = getattr(m, "quant_method", None)
        if isinstance(m, LinearBase) and isinstance(method, UnquantizedLinearMethod):
            saved += _quantize(m)
            m.quant_method = Fp8DenseLinearMethod()
            count += 1
        elif lm_head and isinstance(m, ParallelLMHead) and not isinstance(method, Fp8LMHeadMethod):
            saved += _quantize(m)
            m.quant_method = Fp8LMHeadMethod(m.quant_method)
            count += 1
    torch.cuda.empty_cache()
    logger.info("Dense FP8: converted %d linears, %.2f GiB less to read per step", count, saved / 2**30)


_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _nvfp4_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """w through NVFP4 (16-wide blocks along K, e4m3 block scales, one fp32
    global scale) and back to its dtype, rounding to nearest."""
    N, K = w.shape
    wf = w.float().view(N, K // 16, 16)
    gscale = wf.abs().amax().clamp(min=1e-12) / (6.0 * 448.0)
    bscale = (wf.abs().amax(-1, keepdim=True) / 6.0 / gscale).to(torch.float8_e4m3fn).float()
    bscale = torch.where(bscale == 0, torch.ones_like(bscale), bscale)
    v = wf / (bscale * gscale)
    grid = torch.tensor(_FP4_GRID, device=w.device)
    mid = (grid[1:] + grid[:-1]) / 2
    q = grid[torch.bucketize(v.abs().clamp(max=6.0), mid)] * v.sign()
    return (q * bscale * gscale).view(N, K).to(w.dtype)


@torch.no_grad()
def simulate_nvfp4(model: torch.nn.Module) -> None:
    """Diagnostic: round every eligible bf16 linear (not the lm_head) through
    NVFP4 in place, to measure what 4-bit dense weights would cost in quality."""
    if os.environ.get("VLLM_DENSE_NVFP4_SIM") != "1":
        return
    count = 0
    for name, m in model.named_modules():
        w = getattr(m, "weight", None)
        method = getattr(m, "quant_method", None)
        if (isinstance(m, LinearBase) and isinstance(method, UnquantizedLinearMethod)
                and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16 and w.dim() == 2
                and w.shape[1] % 16 == 0 and not _EXCLUDE.search(name)):
            w.copy_(_nvfp4_roundtrip(w))
            count += 1
    logger.warning("NVFP4 simulation: rounded %d linears through NVFP4", count)
