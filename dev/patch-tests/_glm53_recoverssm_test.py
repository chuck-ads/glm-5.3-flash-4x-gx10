#!/usr/bin/env python3
"""KDA RecoverSSM (experimental/fixes/recoverssm.py) against the stock spec path.

Drives both through the same simulated speculative decode at GLM-5.3-Flash's
per-rank KDA shapes (TP=4: 16 heads x 128, conv width 4), several layers per
KV group, with batch-level k varying step to step (2, 3, 4, 5, 7) and each
row accepting 0..k drafts.

  stock:      causal_conv1d_update(num_accepted_tokens = last step's count)
              + fused_recurrent_kda with 1 + k state slots per request
  recoverssm: the same conv call with num_accepted_tokens = 1
              + kda_recoverssm_verify from one checkpoint block
              + Glm5NextRecoverSSMCommitContext.commit(num_sampled)

Every step, per layer, bit for bit:
  - conv output and recurrent output for every verified token
  - the checkpoint is unchanged by the verify
  - after the commit: the recurrent state equals the stock slot n - 1, and the
    conv window at column 0 equals the stock window at column n - 1
  - align mode (prefix caching): the state is committed to the block the
    runner will read next, and a step that crosses a block boundary also
    leaves the stock state after the boundary token in the boundary block
Runs with and without request_indices (spec rows interleaved with other rows)
and with a trailing cudagraph padding row.

Pass: everything bit-identical. The summary counts checks and failures per
check class, with the first failing step (later failures in a class can be
fallout: the two paths then carry different states). RECOVERSSM_TEST_ATOL=<x>
accepts differences up to x and reports them (diagnosis only).
--drift N runs one chained config for N steps and prints how far the
committed state and the outputs move from the stock path's.

No weights needed. The module is found as vllm.models.glm5next.common.recoverssm
when recoverssm.yaml mounts it, else RECOVERSSM_PY, else next to this file,
else ../../experimental/fixes/recoverssm.py. On one GB10:

    mkdir -p /tmp/rs && cp dev/patch-tests/_glm53_recoverssm_test.py \\
        experimental/fixes/recoverssm.py /tmp/rs/
    sudo docker run --rm --gpus all --ipc=host -v /tmp/rs:/rs -w / \\
        --entrypoint python3 <glm53 image> /rs/_glm53_recoverssm_test.py [--bench]

or `docker cp` both files into a glm53 container and `docker exec` python3 on
the test. It allocates up to ~0.7 GB of GPU memory: run it on a box that is
not serving, since a serving GB10 has 1.5-3 GiB free.
--bench adds a timing of the stock spec kernel against verify + commit at
34 layers.
"""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda


def _load_module():
    name = "vllm.models.glm5next.common.recoverssm"
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    here = Path(__file__).resolve().parent
    candidates = [
        os.environ.get("RECOVERSSM_PY"),
        here / "recoverssm.py",
        here.parent.parent / "experimental" / "fixes" / "recoverssm.py",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            print(f"recoverssm module: {path}")
            return module
    raise SystemExit("recoverssm.py not found; set RECOVERSSM_PY")


rs = _load_module()

dev = torch.device("cuda")
H, D = 16, 128  # per rank at TP=4 (64 heads)
PROJ = H * D
CONV_DIM = 3 * PROJ
WIDTH = 4
LOWER_BOUND = -5.0
DTYPE = torch.bfloat16
LEVELS = (2, 3, 4, 5, 7)
ATOL = float(os.environ.get("RECOVERSSM_TEST_ATOL", "0"))
DS = is_conv_state_dim_first()


CLASSES = (
    "conv output",
    "recurrent output",
    "checkpoint untouched",
    "committed state",
    "committed conv",
    "boundary state",
    "boundary conv",
)


class Checker:
    def __init__(self) -> None:
        self.checks = dict.fromkeys(CLASSES, 0)
        self.failures = dict.fromkeys(CLASSES, 0)
        self.inexact = dict.fromkeys(CLASSES, 0)
        self.worst = dict.fromkeys(CLASSES, 0.0)
        self.first: dict[str, str] = {}
        self.printed = 0

    def eq(self, cls: str, where: str, got: torch.Tensor, want: torch.Tensor) -> None:
        self.checks[cls] += 1
        if torch.equal(got, want):
            return
        diff = (got.float() - want.float()).abs()
        err = float(diff.nan_to_num(float("inf")).max())
        self.worst[cls] = max(self.worst[cls], err)
        if err <= ATOL:
            self.inexact[cls] += 1
            return
        self.failures[cls] += 1
        self.first.setdefault(cls, where)
        if self.printed < 10:
            self.printed += 1
            scale = float(want.float().abs().max())
            print(f"  MISMATCH {cls} at {where}: max abs {err:.3g} (scale "
                  f"{scale:.3g}), {int((diff > 0).sum())} of {diff.numel()} differ")

    def report(self) -> None:
        print(f"{'class':<22}{'checks':>8}{'fail':>8}{'inexact':>9}{'worst':>11}  first failure")
        for cls in CLASSES:
            print(f"{cls:<22}{self.checks[cls]:>8}{self.failures[cls]:>8}"
                  f"{self.inexact[cls]:>9}{self.worst[cls]:>11.3g}  {self.first.get(cls, '-')}")


def conv_pool(lines: int, state_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(raw kv_cache[0] tensor, its (lines, dim, state_len) view for the kernels)."""
    if DS:
        raw = torch.zeros(lines, CONV_DIM, state_len, device=dev, dtype=DTYPE)
        return raw, raw
    raw = torch.zeros(lines, state_len, CONV_DIM, device=dev, dtype=DTYPE)
    return raw, raw.transpose(-1, -2)


def make_layer(g: torch.Generator, lines_stock: int, lines_rs: int, k: int, rows: int):
    state_len = WIDTH - 1 + k
    layer = SimpleNamespace()
    layer.conv_w = (torch.randn(CONV_DIM, WIDTH, device=dev, generator=g) * 0.3).float()
    layer.A_log = torch.log(
        torch.rand(1, 1, H, 1, device=dev, generator=g) * 8 + 1
    ).float()
    layer.dt_bias = (torch.randn(PROJ, device=dev, generator=g) * 0.5).float()
    layer.kda_lower_bound = LOWER_BOUND
    layer.conv_s_raw, layer.conv_s = conv_pool(lines_stock, state_len)
    layer.rec_s = torch.zeros(lines_stock, H, D, D, device=dev, dtype=torch.float32)
    conv_r_raw, layer.conv_r = conv_pool(lines_rs, state_len)
    layer.rec_r = torch.zeros(lines_rs, H, D, D, device=dev, dtype=torch.float32)
    layer.kv_cache = (conv_r_raw, layer.rec_r)
    layer.recoverssm_records = rs.allocate_records(rows, H, D, k + 1, DTYPE, dev)
    return layer


def stock_step(layer, x, g1, beta, qsl, cols, na_prev, k):
    y = causal_conv1d_update(
        x.clone(), layer.conv_s, layer.conv_w, None, activation="silu",
        conv_state_indices=cols[:, 0], num_accepted_tokens=na_prev,
        query_start_loc=qsl, max_query_len=k + 1,
    )
    q, kk, v = y.split(PROJ, dim=-1)
    o, _ = fused_recurrent_kda(
        q=q.reshape(1, -1, H, D), k=kk.reshape(1, -1, H, D), v=v.reshape(1, -1, H, D),
        g=g1, beta=beta, initial_state=layer.rec_s, use_qk_l2norm_in_kernel=True,
        cu_seqlens=qsl, ssm_state_indices=cols, num_accepted_tokens=na_prev,
        sigmoid_beta=True, a_log=layer.A_log, g_bias=layer.dt_bias,
        compute_gate=True, lower_bound=LOWER_BOUND,
    )
    return y, o


def rs_step(layer, x, g1, beta, qsl, idx, ones, k):
    y = causal_conv1d_update(
        x.clone(), layer.conv_r, layer.conv_w, None, activation="silu",
        conv_state_indices=idx, num_accepted_tokens=ones,
        query_start_loc=qsl, max_query_len=k + 1,
    )
    q, kk, v = y.split(PROJ, dim=-1)
    o = rs.kda_recoverssm_verify(
        q=q.reshape(1, -1, H, D), k=kk.reshape(1, -1, H, D), v=v.reshape(1, -1, H, D),
        g=g1, beta=beta, a_log=layer.A_log, g_bias=layer.dt_bias,
        lower_bound=LOWER_BOUND, checkpoint_state=layer.rec_r,
        records=layer.recoverssm_records, query_start_loc=qsl, state_indices=idx,
    )
    return y, o


def run(k: int, align: bool, seed: int, chk: Checker, steps: int = 40,
        num_reqs: int = 6, num_layers: int = 3, block_size: int = 16,
        drift: list | None = None) -> None:
    g = torch.Generator(device=dev).manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    B = num_reqs
    rows = B + 1  # one trailing padding row, as a FULL-graph batch has
    width = 2 + (steps * (k + 1)) // block_size + 2
    lines_stock = 1 + B * (k + 1)
    lines_rs = 1 + B * (width if align else 1)
    layers = [make_layer(g, lines_stock, lines_rs, k, rows) for _ in range(num_layers)]
    ctx = rs.Glm5NextRecoverSSMCommitContext.create(
        layers, spec_query_len=k + 1, max_num_reqs=rows
    )

    stock_cols = torch.zeros(rows, k + 1, dtype=torch.int32)
    for i in range(B):
        stock_cols[i] = torch.arange(1 + i * (k + 1), 1 + (i + 1) * (k + 1))
    stock_cols = stock_cols.to(dev)
    if align:
        block_table = torch.zeros(rows, width, dtype=torch.int32)
        for i in range(B):
            block_table[i] = torch.arange(1 + i * width, 1 + (i + 1) * width)
        block_table = block_table.to(dev)
        num_computed = torch.randint(1, 2 * block_size, (B,), generator=cpu)
        cur_col = ((num_computed - 1) // block_size).tolist()
    else:
        block_table = None
        num_computed = torch.zeros(B, dtype=torch.int64)
        cur_col = [0] * B

    def rs_block(i: int, col: int) -> int:
        return int(block_table[i, col]) if align else 1 + i

    # Same random checkpoint in both paths: stock slot 0, read with na = 1.
    for layer in layers:
        for i in range(B):
            state = torch.randn(H, D, D, device=dev, generator=g) * 0.1
            hist = torch.randn(CONV_DIM, WIDTH - 1, device=dev, generator=g).to(DTYPE)
            layer.rec_s[int(stock_cols[i, 0])] = state
            layer.conv_s[int(stock_cols[i, 0]), :, : WIDTH - 1] = hist
            b = rs_block(i, cur_col[i])
            layer.rec_r[b] = state
            layer.conv_r[b, :, : WIDTH - 1] = hist
    na_prev = torch.ones(rows, dtype=torch.int32, device=dev)
    ones = torch.ones(rows, dtype=torch.int32, device=dev)

    for step in range(steps):
        k_step = min(LEVELS[int(torch.randint(len(LEVELS), (1,), generator=cpu))], k)
        # Most rows verify k_step drafts. About one in five has no drafts.
        qlens = [
            1 if torch.rand(1, generator=cpu) < 0.2 else k_step + 1 for _ in range(B)
        ] + [0]
        qsl_cpu = torch.zeros(rows + 1, dtype=torch.int32)
        qsl_cpu[1:] = torch.cumsum(torch.tensor(qlens), 0)
        qsl = qsl_cpu.to(dev)
        T = int(qsl_cpu[-1])
        x = torch.randn(T, CONV_DIM, device=dev, generator=g).to(DTYPE)
        g1 = torch.randn(1, T, H, D, device=dev, generator=g).to(DTYPE)
        # beta as a column slice of a wider projection, like kda.py's (its row
        # is 6416 wide: a multiple of 16, as here, which Triton specializes on).
        beta = torch.randn(1, T, H + 16, device=dev, generator=g).to(DTYPE)[..., :H]

        # Runner align preprocess: the running block moves to the column the
        # verify reads, (seq_len - 1) // block_size.
        if align:
            for i in range(B):
                dst = (int(num_computed[i]) + qlens[i] - 1) // block_size
                if dst != cur_col[i]:
                    src_b, dst_b = rs_block(i, cur_col[i]), rs_block(i, dst)
                    for layer in layers:
                        layer.rec_r[dst_b] = layer.rec_r[src_b]
                        layer.conv_r[dst_b] = layer.conv_r[src_b]
                    cur_col[i] = dst
        idx = torch.tensor(
            [rs_block(i, cur_col[i]) for i in range(B)] + [0],
            dtype=torch.int32, device=dev,
        )

        for li, layer in enumerate(layers):
            before = layer.rec_r[idx[:B].long()].clone()
            y_s, o_s = stock_step(layer, x, g1, beta, qsl, stock_cols, na_prev, k)
            y_r, o_r = rs_step(layer, x, g1, beta, qsl, idx, ones, k)
            tag = f"k={k} align={align} seed={seed} step={step} layer={li}"
            chk.eq("conv output", tag, y_r[:T], y_s[:T])
            chk.eq("recurrent output", tag, o_r[:, :T], o_s[:, :T])
            chk.eq("checkpoint untouched", tag, layer.rec_r[idx[:B].long()], before)
            if drift is not None:
                out_err = float((o_r[:, :T].float() - o_s[:, :T].float()).abs().max())

        accepted = [int(torch.randint(0, q, (1,), generator=cpu)) for q in qlens[:B]]
        n_sampled = [a + 1 for a in accepted]

        # Odd steps: spec rows interleaved with other batch rows.
        interleave = step % 2 == 1
        if interleave:
            batch_rows = [2 * i + 1 for i in range(B)]
            nb = 2 * B + 1
            request_indices = torch.tensor(batch_rows, dtype=torch.int32, device=dev)
        else:
            batch_rows = list(range(B))
            nb = B
            request_indices = None
        num_sampled = torch.randint(0, 9, (nb,), generator=cpu, dtype=torch.int32)
        nc_batch = torch.randint(0, 99, (nb,), generator=cpu, dtype=torch.int32)
        bt_batch = torch.zeros(nb, width if align else 1, dtype=torch.int32)
        for i, r in enumerate(batch_rows):
            num_sampled[r] = n_sampled[i]
            nc_batch[r] = int(num_computed[i])
            if align:
                bt_batch[r] = block_table[i].cpu()
        align_kwargs = {}
        if align:
            align_kwargs = dict(
                block_table=bt_batch.to(dev),
                num_computed_tokens=nc_batch.to(dev),
                mamba_block_size=block_size,
            )
        ctx.commit(
            num_sampled.to(dev), idx[:B], qsl[: B + 1],
            request_indices=request_indices, **align_kwargs,
        )

        state_err = 0.0
        for i in range(B):
            n = n_sampled[i]
            nc = int(num_computed[i])
            final_col = (nc + n) // block_size if align else 0
            final_b = rs_block(i, final_col)
            for li, layer in enumerate(layers):
                tag = f"k={k} align={align} seed={seed} step={step} layer={li} req={i} n={n}"
                s_col = int(stock_cols[i, n - 1])
                c_col = int(stock_cols[i, 0])
                chk.eq("committed state", tag, layer.rec_r[final_b], layer.rec_s[s_col])
                chk.eq("committed conv", tag,
                       layer.conv_r[final_b, :, : WIDTH - 1],
                       layer.conv_s[c_col, :, n - 1 : n - 1 + WIDTH - 1])
                if drift is not None:
                    want = layer.rec_s[s_col]
                    state_err = max(state_err, float(
                        (layer.rec_r[final_b] - want).abs().max() / want.abs().max()))
                next_boundary = (nc // block_size + 1) * block_size
                if align and nc + n >= next_boundary:
                    blen = next_boundary - nc
                    bb = rs_block(i, next_boundary // block_size - 1)
                    chk.eq("boundary state", tag,
                           layer.rec_r[bb], layer.rec_s[int(stock_cols[i, blen - 1])])
                    chk.eq("boundary conv", tag,
                           layer.conv_r[bb, :, : WIDTH - 1],
                           layer.conv_s[c_col, :, blen - 1 : blen - 1 + WIDTH - 1])
            if align:
                num_computed[i] = nc + n
                cur_col[i] = final_col
        na_prev = torch.tensor(n_sampled + [1], dtype=torch.int32, device=dev)
        if drift is not None:
            drift.append((step, state_err, out_err))
    torch.cuda.synchronize()


def drift_run(steps: int) -> None:
    """Chained, never resynced: how far RecoverSSM moves from the stock path."""
    samples: list = []
    run(7, False, 3, Checker(), steps=steps, num_layers=1, drift=samples)
    every = max(1, steps // 10)
    print(f"drift over {steps} steps (k=7, 6 requests; state error relative to "
          "the state's max, output error absolute):")
    worst_state = worst_out = 0.0
    for step, state_err, out_err in samples:
        worst_state = max(worst_state, state_err)
        worst_out = max(worst_out, out_err)
        if (step + 1) % every == 0:
            print(f"  step {step + 1:>6}: state {state_err:.3g} (worst so far "
                  f"{worst_state:.3g}), last-layer output {out_err:.3g} "
                  f"(worst {worst_out:.3g})")


def bench(num_reqs: int, k: int = 7, num_layers: int = 34, iters: int = 20) -> None:
    """Stock spec kernel vs verify + commit, all rows verifying k drafts.

    Layers share one pool here (timing only), so the numbers are per step for
    num_layers layers of one KV group.
    """
    g = torch.Generator(device=dev).manual_seed(0)
    q_len = k + 1
    rows = num_reqs
    layer = make_layer(g, 1 + rows * q_len, 1 + rows, k, rows)
    layer.rec_s.normal_(generator=g).mul_(0.1)
    layer.rec_r.normal_(generator=g).mul_(0.1)
    layers = [layer] * num_layers
    ctx = rs.Glm5NextRecoverSSMCommitContext.create(
        layers, spec_query_len=q_len, max_num_reqs=rows
    )
    T = rows * q_len
    qsl = torch.arange(0, T + 1, q_len, dtype=torch.int32, device=dev)
    cols = (1 + torch.arange(rows * q_len, dtype=torch.int32, device=dev)).view(rows, q_len)
    idx = 1 + torch.arange(rows, dtype=torch.int32, device=dev)
    na = torch.full((rows,), q_len, dtype=torch.int32, device=dev)
    y = torch.randn(T, CONV_DIM, device=dev, generator=g).to(DTYPE)
    q, kk, v = (t.reshape(1, -1, H, D) for t in y.split(PROJ, dim=-1))
    g1 = torch.randn(1, T, H, D, device=dev, generator=g).to(DTYPE)
    beta = torch.randn(1, T, H, device=dev, generator=g).to(DTYPE)
    out = torch.empty(1, T, H, D, device=dev, dtype=DTYPE)
    sampled = torch.full((rows,), q_len, dtype=torch.int32, device=dev)

    def stock():
        for _ in range(num_layers):
            fused_recurrent_kda(
                q=q, k=kk, v=v, g=g1, beta=beta, initial_state=layer.rec_s,
                use_qk_l2norm_in_kernel=True, cu_seqlens=qsl, ssm_state_indices=cols,
                num_accepted_tokens=na, out=out, sigmoid_beta=True, a_log=layer.A_log,
                g_bias=layer.dt_bias, compute_gate=True, lower_bound=LOWER_BOUND,
            )

    def recover():
        for _ in range(num_layers):
            rs.kda_recoverssm_verify(
                q=q, k=kk, v=v, g=g1, beta=beta, a_log=layer.A_log,
                g_bias=layer.dt_bias, lower_bound=LOWER_BOUND,
                checkpoint_state=layer.rec_r, records=layer.recoverssm_records,
                query_start_loc=qsl, state_indices=idx, out=out,
            )
        ctx.commit(sampled, idx, qsl)

    def verify_only():
        for _ in range(num_layers):
            rs.kda_recoverssm_verify(
                q=q, k=kk, v=v, g=g1, beta=beta, a_log=layer.A_log,
                g_bias=layer.dt_bias, lower_bound=LOWER_BOUND,
                checkpoint_state=layer.rec_r, records=layer.recoverssm_records,
                query_start_loc=qsl, state_indices=idx, out=out,
            )

    def timed(fn) -> float:
        fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    t_stock, t_rec, t_ver = timed(stock), timed(recover), timed(verify_only)
    print(f"bench rows={rows} k={k} layers={num_layers}: stock {t_stock:.2f} ms, "
          f"verify {t_ver:.2f} ms + commit {t_rec - t_ver:.2f} ms = {t_rec:.2f} ms")


chk = Checker()
for k in (7, 3):
    for align in (False, True):
        for seed in (0, 1):
            run(k, align, seed, chk)
            print(f"k={k} align={align} seed={seed}: {sum(chk.checks.values())} "
                  f"checks so far, {sum(chk.failures.values())} failures")
chk.report()
if "--bench" in sys.argv:
    for n in (16, 32, 64):
        bench(n)
if "--drift" in sys.argv:
    drift_run(int(sys.argv[sys.argv.index("--drift") + 1]))
checks, failures = sum(chk.checks.values()), sum(chk.failures.values())
inexact = sum(chk.inexact.values())
if failures:
    print(f"RECOVERSSM: FAIL ({failures} of {checks} checks)")
    sys.exit(1)
if inexact:
    print(f"RECOVERSSM: PASS within atol {ATOL} ({inexact} of {checks} "
          "not bit-identical)")
else:
    print(f"RECOVERSSM: PASS, bit-identical ({checks} checks)")
