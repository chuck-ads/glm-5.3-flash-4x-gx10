# Experimental speedups

Compose overrides that stack on `compose/glm53.yaml`. Each one mounts files
over the v8 image (vLLM nightly ddd6fbca); several replace whole vLLM or
FlashInfer files, so they only match that image.

```
docker compose -f compose/glm53.yaml \
  -f experimental/compose/arx.yaml -f experimental/compose/snapshot.yaml \
  -f experimental/compose/adaptive-k.yaml -f experimental/compose/fp8.yaml \
  -f experimental/compose/megamoe.yaml -f experimental/compose/fixes.yaml \
  -f experimental/compose/sp.yaml up -d
```

## Results

Single stream, thinking off, 512 tokens (`dev/repro/decode.py` prompts):

| | structured | code | prose |
|---|---|---|---|
| v8, DFlash2 k=7 | 121.9 | 91.3 | 38.7 |
| all overrides | 166.9 | 118.1 | 63.7 |

Concurrent streams, aggregate tok/s at 1/2/4/8 streams: v8 85.5/65.4/99.5/147.8,
all overrides 122.5/99.8/139.3/189.8. RigMark (reasoning=low) code / prose /
structured 101.2 / 59.8 / 153.9 tok/s, 64k cold prefill 4,523 tok/s. Needle
retrieval 12/12 up to 507k tokens.

Cold prefill, tok/s (random-word prompts, nothing cached):

| | 32k | 128k |
|---|---|---|
| v8 | 2,730 | 2,679 |
| all overrides | 4,521 | 4,385 |

Boot goes from about 8 minutes to about 3.5 once snapshots exist.

## The pieces

- **arx** (`arx/`): all-reduce over RDMA for tensors up to 256 KB, without
  NCCL. It takes 13-27 us where NCCL takes 80-90. The GPU publishes a
  sequence number, and a CPU thread posts the RDMA writes over both ConnectX
  roots.
- **snapshot** (`snapshot/weight_snapshot.py`, `base_loader.py`): the first
  boot writes each rank's processed weights to disk. Later boots build the
  model on dummy weights through the normal path, then overwrite every tensor
  in place (about 15 s for 46 GiB). `modelopt.py` and
  `flashinfer_cutlass_moe.py` are left from an earlier design and may no
  longer be needed.
- **fp8** (`snapshot/dense_fp8.py`): the NVFP4 checkpoint leaves about 4 GB
  per rank of dense linears in bf16, and decode reads them every step. This
  converts them to FP8 after load (per-channel weight scales, per-token
  activation scales, CUTLASS scaled_mm). Prefill-sized batches go through in
  2048-row pieces when the weight is over 16 MiB: CUTLASS rereads the whole
  weight for every row of tiles, and past L2 that comes from DRAM, so one call
  on a 16k x 6416 x 4096 GEMM runs at 68 TFLOPS and the pieces at 169. +14-20% decode, and about +0.5% NLL
  on prose. Layers matching `VLLM_DENSE_W4` (by default the KDA in_proj and
  every drafter layer, attention o_proj and the shared experts) go to NVFP4 instead, through `megamoe/megadense4.cu`
  (W4A16 at the 4-bit roofline, about twice as fast as the FP8 GEMM). They
  keep an FP8 copy for batches of more than 32 tokens. That is +6-10% decode.
  The in_proj costs about as much NLL again as FP8 did. The drafter layers only
  change acceptance. NVFP4 on every dense layer cost 2-3x more NLL, so it is
  not the default.
- **megamoe** (`megamoe/`): an NVFP4 MoE kernel for batches of up to 8 tokens.
  It reads the CUTLASS backend's own tensors, so CUTLASS still handles
  prefill. Activations stay 16-bit, which is exact in the weights, where
  CUTLASS rounds them to FP4. With `VLLM_MOE_PREFILL=1` (the default here),
  batches of 1024+ tokens take `megamoe/moe_prefill.cu`: the same W4A4 math as
  CUTLASS on the same tensors, with the token gather, SwiGLU and FP4
  requantization folded into the fc1 GEMM (hand-written block-scaled
  `mma.sync` NVFP4). About 18 ms per layer at 16k tokens against CUTLASS's
  22-26, equally close to an fp32 reference. +7-8% prefill.
- **adaptive-k** (`adaptive-k/adaptive_k.py`): a scheduler that picks how many
  of the 7 DFlash2 drafts each step verifies (2, 3, 4, 5 or 7). It uses recent
  per-position acceptance and a measured cost per level
  (`VLLM_ADAPTIVE_K_COST_MS`). The drafter uses Triton attention, which avoids
  a mid-step host sync.
- **fixes** (`fixes/`): things found by profiling.
  - FlashInfer's MLA planner cloned a 136 MB metadata buffer on every step,
    only to roll it back on error.
  - The GLM indexer's head gate ran a fp32 GEMM that took 69 us per layer.
  - FlashInfer's sparse MLA runs at about 10 TFLOPS on GB10 whatever the
    indices. `gb10_sparse_mla.py` is a Triton kernel for GLM's NoPE MLA (one
    program per query, all 16 heads, 32 gathered keys per block) that runs
    3.3x faster. `VLLM_TRITON_SPARSE_MLA=0` goes back to FlashInfer.
- **sp** (`fixes/model.py`): sequence parallelism for prefill. A forward of
  at least 1024 tokens keeps the residual stream split across the TP ranks, so
  mHC and the norms run on a quarter of the tokens, with an all-gather before
  and a reduce-scatter after attention and the MLP. +17% prefill. Decode stays
  plain TP: applied to every batch, SP cost decode 10-15%. The KDA layers'
  attention inputs are gathered as per-token FP8, half the bytes, since
  in_proj's FP8 GEMM would quantize the same rows the same way (+2%).
- **arxbig** (`arx/arxbig.cu`, `VLLM_ARXBIG`, `VLLM_ARXBIG_RS`): all-gather and
  reduce-scatter over RoCE for prefill-sized SP collectives, from pinned
  buffers the ConnectX writes directly. The all-gather is ~10% faster than
  NCCL's (187 vs 165 Gb/s), which alone is worth <1% of prefill; the point is
  the reduce-scatter. With `VLLM_GLM_SP_MOE_FUSED`, MoE layers under SP run the
  router, the shared expert and moe_prefill's fc1/fc2 in model.py, and one
  kernel writes shared + scaled routed sum (fp32, rounded once) straight into
  the send buffer, publishing rows as it goes, so the network runs under the
  finalize. It replaces the runner's scale and add passes and NCCL's
  reduce-scatter: ~12.5 -> ~8.4 ms per MoE layer at 16k tokens, prefill +2.5-3%.
  All ranks write to all peers at once and this fabric has no PFC, so
  incast drops packets and go-back-N retransmits make individual calls vary
  (4-9 ms); per-destination serialization was worse (two QPs cannot fill a
  link).
- **arx prefetch** (`VLLM_GLM_ARX_PREFETCH`): while a decode all-reduce waits
  for its peers, its threads ask L2 for the weights the next kernels read (the
  router gate and shared expert after attention, the next in_proj after the
  MoE). About -0.6 ms per 32 ms step.

`dense_fp8.simulate_nvfp4` is a diagnostic. It rounds the dense weights
through NVFP4 to measure the quality cost: about +1% NLL on prose.
