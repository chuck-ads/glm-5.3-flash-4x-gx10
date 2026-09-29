# TP=3 (three boxes)

GLM-5.3-Flash at tensor parallel 3. Four dimensions do not divide by 3, so
they are zero-padded; the padding is exact (zero heads and zero expert
columns add nothing) and the checkpoint on disk is not rewritten.

| | stock | padded | per rank |
|---|---|---|---|
| MLA heads | 64 | 66 | 22 |
| KDA heads | 64 | 66 | 22 |
| MoE intermediate | 2048 | 2304 | 768 |
| vocab | 154880 | 154944 | 51648 |
| DFlash2 drafter q / kv heads | 32 / 8 | 36 / 9 | 12 / 3 |

## Pieces

- `make_padded.py SRC_MODEL DST_MODEL SRC_DRAFT DST_DRAFT`: the target dir is
  hardlinks of every checkpoint file plus a `config.json` with the padded sizes
  and `tp_pad_orig`; the drafter (small) is padded offline. Run it inside the image.
- `tp3pad.py`: pads the target's tensors as `load_weights` reads them (hooked
  from `experimental/fixes/model.py`; a no-op when the config has no `tp_pad_orig`).
- `vocab_parallel_embedding.py`: pads the vocab on to a multiple of 64 x TP.
- `experimental/fixes/gb10_sparse_mla.py`: the Triton sparse-MLA kernel took
  `tl.arange(0, H)` over heads, which needs a power of two; 22 heads per rank
  now run in the next power-of-two tile with a head mask.
- `experimental/snapshot/dense_fp8.py`: the KDA in_proj at TP=3 has N=8726
  rows, not a multiple of 16, and stayed bf16; rows are now zero-padded to 16
  and the output trimmed.
- `image/entrypoint.sh`: a `TP=3` case. KV pin 12 GiB (26 GiB had the
  host OOM-killer take rank 0 during graph capture), block size 3456 (vLLM
  lifts it to 3584, 14% mamba page padding against 46% at the 4608 it picks
  from 2304).

## Running it

Everything else is the four-box setup from the main README (image, mentatd,
fabric), on three boxes.

1. **Padded model dirs, on every box.** The target dir is hardlinks, so it must
   sit on the same filesystem as the checkpoint; the drafter is copied and
   padded. Run inside the image this repo builds:

       docker run --rm -v /srv/models:/srv/models --entrypoint python3 \
         -v "$PWD/experimental/tp3":/tp3:ro spark-glm53:v9 /tp3/make_padded.py \
         /srv/models/glm-5.3-flash-nvfp4 /srv/models/glm-5.3-flash-nvfp4-tp3 \
         /srv/models/glm-5.3-flash-dflash2 /srv/models/glm-5.3-flash-dflash2-tp3

2. **compose/.env on every box** (tp3.yaml sets `TP=3`):

       MODEL_HOST_DIR=/srv/models/glm-5.3-flash-nvfp4-tp3
       DFLASH_HOST_DIR=/srv/models/glm-5.3-flash-dflash2-tp3

3. **Start** with every override, `recoverssm.yaml` included, and `tp3.yaml` last (it
   restates adaptive-k.yaml's `EXTRA_ARGS` and adds a data-parallel vision
   tower, whose 16 heads do not split by 3):

       docker compose -f compose/glm53.yaml \
         -f experimental/compose/arx.yaml -f experimental/compose/snapshot.yaml \
         -f experimental/compose/adaptive-k.yaml -f experimental/compose/fp8.yaml \
         -f experimental/compose/megamoe.yaml -f experimental/compose/fixes.yaml \
         -f experimental/compose/sp.yaml -f experimental/compose/recoverssm.yaml \
         -f experimental/compose/tp3.yaml up -d

   The first boot pads while it
   loads and writes a weight snapshot under its own tag (`tp3pad-...`, set in
   tp3.yaml; a TP=4 snapshot must never restore here); later boots restore it.

The entrypoint's `TP=3` case sets the KV pin (12 GiB), block size (3456),
`MAX_NUM_SEQS` 64 with RecoverSSM (32 without) and the adaptive-k starting cost. With RecoverSSM on, the head keeps ~6.6 GB of
host memory free while serving; a 16 GiB pin would leave ~2.6 GB, so do not raise it.

## Tried and not kept

At TP=3, on this stack, one change at a time against the defaults above:
sequence parallel off (-16% prefill), `VLLM_ARXBIG_AG=1` (-26 to -30%
prefill), `max_num_batched_tokens` 8192 (-4% at 22k), megamoe
`VLLM_MEGAMOE_MAX_TOKENS=16` and its tile variants (within noise), 72 MLA
heads instead of 66 (-10% prose decode, measured on the PR #4 base). A MoE intermediate of 2112 (704 per
rank) would pad less, but megamoe needs a multiple of 128 per rank.

## Tests

`dev/patch-tests/_glm53_tp3pad_test.py` (CPU) and
`dev/patch-tests/_glm53_tp3_sparse_mla_test.py` (one GB10); each file says how
to run it.

## Measured

Two sets of three boxes (one 200G switch, both ConnectX ports), each booted
clean with the defaults above (RecoverSSM on, `MAX_NUM_SEQS` 64), on this
branch over main 6db84d4 plus the capture-size fix from its own PR
(`tp-cudagraph-tiers`: an explicit capture cap dropped adaptive-k's decode
sizes 3, 5, 6, 10, 12 and 15). KV holds 1.48M tokens.

| | set 1 | set 2 |
|---|---|---|
| RigMark prose / code / structured, tok/s | 43.8 / 75.7 / 113.8 | 44.0 / 76.2 / 114.0 |
| RigMark cold prefill 8k / 32k / 64k, tok/s | 3295 / 3602 / 3565 | 3325 / 3610 / 3583 |
| `gate/prefill.py` cold 32k / 128k, tok/s (2 seeds) | 3574, 3533 / 3440, 3449 | 3561, 3554 / 3422, 3437 |
| `gate/conc_workload.py code` 1 / 2 / 4 / 8 streams, tok/s | 86.5 / 101.8 / 137.4 / 163.7 | 87.4 / 100.5 / 153.7 / 170.2 |
| `conc_workload.py mixed` 8 / 16 / 32 / 48 streams, tok/s | 134.1 / 195.5 / 262.7 / 335.9 | 141.0 / 200.3 / 260.0 / 333.9 |
| `dev/repro/needle.py` 32k to 480k | 12/12 | 12/12 |

At 48 streams, `MAX_NUM_SEQS` 64 gave +24% over 32 (269 tok/s); single-stream
decode is the same at both. A 30-minute soak of the same TP=3 changes on main
14cc721 (`MAX_NUM_SEQS` 32, 16 mixed streams back to back, 824 requests) had
no errors or restarts.
