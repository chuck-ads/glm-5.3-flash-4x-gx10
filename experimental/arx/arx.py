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


# (address, bytes) ranges for the next all-reduce to prefetch into L2 while it waits.
_prefetch: tuple = ()


def set_prefetch(tensors) -> None:
    """The next arx all-reduce asks L2 for these tensors' bytes while it waits for peers."""
    global _prefetch
    _prefetch = tuple((t.data_ptr(), t.numel() * t.element_size()) for t in tensors if t is not None)


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
        global _prefetch
        out = torch.empty_like(t)
        pf, _prefetch = _prefetch, ()
        self.ext.allreduce(t, out, [p for p, _ in pf], [b for _, b in pf])
        return out


_big_ext = None


class ArxBig:
    """Prefill-sized all-gathers over RoCE (arxbig.cu): ~10% faster than NCCL's.

    Enabled with VLLM_ARXBIG=1 next to arx. all_gather returns a view of a
    pinned buffer that the fourth all-gather after it overwrites, so a caller
    that keeps the result longer must copy it.
    """

    def __init__(self, group: ProcessGroup, device: torch.device):
        global _big_ext
        self.disabled = True
        self.rs = False
        hcas = [h.strip("=^").split(":")[0] for h in os.environ.get("NCCL_IB_HCA", "").split(",") if h]
        if len(hcas) != 2:
            logger.warning("arxbig needs two RDMA devices in NCCL_IB_HCA, got %r; using NCCL", hcas)
            return
        gid = int(os.environ.get("NCCL_IB_GID_INDEX", "5"))
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        slot = int(os.environ.get("VLLM_ARXBIG_SLOT_MB", "132")) << 20
        slot -= slot % (world * 16)
        with torch.cuda.device(device):
            if _big_ext is None:
                from torch.utils.cpp_extension import load

                _big_ext = load("arxbig", [os.path.join(os.path.dirname(__file__), "arxbig.cu")],
                                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_121a,code=sm_121a"],
                                extra_ldflags=["-libverbs"])
            self.rs = os.environ.get("VLLM_ARXBIG_RS") == "1"
            info = _big_ext.prepare(rank, world, hcas[0], hcas[1], gid, slot, slot if self.rs else 0)
            infos: list[bytes] = [b""] * world
            dist.all_gather_object(infos, info, group=group)
            _big_ext.connect(infos)
            dist.barrier(group=group)
        self.ext, self.world, self.slot = _big_ext, world, slot
        self.min_bytes = int(os.environ.get("VLLM_ARXBIG_MIN_KB", "1024")) << 10
        self.disabled = False
        logger.info("arxbig all-gather: rank %d/%d, %d MiB slots", rank, world, slot >> 20)

    def should_gather(self, t: torch.Tensor) -> bool:
        total = t.numel() * t.element_size() * self.world
        return (t.is_cuda and t.is_contiguous() and t.dim() >= 1 and (t.numel() * t.element_size()) % 16 == 0
                and self.min_bytes <= total <= self.slot)

    def all_gather(self, t: torch.Tensor) -> torch.Tensor:
        return self.ext.all_gather(t)
