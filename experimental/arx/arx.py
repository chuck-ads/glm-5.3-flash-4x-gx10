"""All-reduce over RoCE for GB10 tensor-parallel groups, without NCCL.

NCCL costs about 80-90 us per decode-sized all-reduce on these boxes; this
costs 13 us at 8 KB and 27 us at 64 KB. Each rank writes its partial to its
peers with RDMA over both ConnectX roots (see arx_vllm.cu). Larger all-reduces,
which are prefill-sized, stay on NCCL, whose bandwidth wins there.

Enabled with VLLM_ARX_ALLREDUCE=1. The RDMA devices and GID index come from
NCCL_IB_HCA (exactly two, one per root, in the same subnet order on every
rank) and NCCL_IB_GID_INDEX.
"""
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

_ext = None
_taken = False  # the extension holds one connection per process


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        _ext = load(
            "arx_vllm",
            [os.path.join(os.path.dirname(__file__), "arx_vllm.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_121a,code=sm_121a"],
            extra_ldflags=["-libverbs"],
        )
    return _ext


class ArxCommunicator:
    def __init__(self, group: ProcessGroup, device: torch.device):
        global _taken
        self.disabled = True
        if _taken:
            logger.warning("arx already serves another group in this process; using NCCL here")
            return
        hcas = [h.strip("=^").split(":")[0] for h in os.environ.get("NCCL_IB_HCA", "").split(",") if h]
        if len(hcas) != 2:
            logger.warning("arx needs two RDMA devices in NCCL_IB_HCA, got %r; using NCCL", hcas)
            return
        gid = int(os.environ.get("NCCL_IB_GID_INDEX", "5"))
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        with torch.cuda.device(device):
            ext = _load()
            info = ext.prepare(rank, world, hcas[0], hcas[1], gid)
            infos: list[bytes] = [b""] * world
            dist.all_gather_object(infos, info, group=group)
            ext.connect(infos)
        dist.barrier(group=group)
        _taken = True
        self.ext = ext
        self.max_bytes = min(int(os.environ.get("VLLM_ARX_MAX_BYTES", 256 << 10)), ext.max_bytes())
        self.disabled = False
        logger.info("arx all-reduce: rank %d/%d on %s, gid %d, up to %d bytes", rank, world, hcas, gid, self.max_bytes)

    def should_use(self, t: torch.Tensor) -> bool:
        return (
            t.dtype == torch.bfloat16
            and t.is_cuda
            and t.is_contiguous()
            and t.numel() % 8 == 0
            and t.numel() * 2 <= self.max_bytes
        )

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(t)
        self.ext.allreduce(t, out)
        return out
