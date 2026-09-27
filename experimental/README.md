# Experimental decode speedups

Compose overrides that stack on `compose/glm53.yaml`. Each one mounts files
over the v8 image (vLLM nightly ddd6fbca); several replace whole vLLM or
FlashInfer files, so they only match that image.

```
docker compose -f compose/glm53.yaml \
  -f experimental/compose/arx.yaml -f experimental/compose/snapshot.yaml \
  -f experimental/compose/adaptive-k.yaml -f experimental/compose/fp8.yaml \
  -f experimental/compose/megamoe.yaml -f experimental/compose/fixes.yaml up -d
```

## Results

Single stream, thinking off, 512 tokens (`dev/repro/decode.py` prompts):

| | structured | code | prose |
|---|---|---|---|
| v8, DFlash2 k=7 | 121.9 | 91.3 | 38.7 |
| all overrides | 152.9 | 109.9 | 58.3 |

Boot goes from about 8 minutes to about 3.5 once snapshots exist. Prefill is
unchanged (about 2,650 tok/s at 32k and 128k).

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
  activation scales, CUTLASS scaled_mm). +14-20% decode, and about +0.5% NLL
  on prose.
- **megamoe** (`megamoe/`): an NVFP4 MoE kernel for batches of up to 8 tokens.
  It reads the CUTLASS backend's own tensors, so CUTLASS still handles
  prefill. Activations stay 16-bit, which is exact in the weights, where
  CUTLASS rounds them to FP4.
- **adaptive-k** (`adaptive-k/adaptive_k.py`): a scheduler that picks how many
  of the 7 DFlash2 drafts each step verifies (2, 3, 4, 5 or 7). It uses recent
  per-position acceptance and a measured cost per level
  (`VLLM_ADAPTIVE_K_COST_MS`). The drafter uses Triton attention, which avoids
  a mid-step host sync.
- **fixes** (`fixes/`): two things found by profiling.
  - FlashInfer's MLA planner cloned a 136 MB metadata buffer on every step,
    only to roll it back on error.
  - The GLM indexer's head gate ran a fp32 GEMM that took 69 us per layer.

`dense_fp8.simulate_nvfp4` is a diagnostic. It rounds the dense weights
through NVFP4 to measure the quality cost: about +1% NLL on prose.
