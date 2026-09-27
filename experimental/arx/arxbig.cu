// arxbig: prefill-sized all-gather and reduce-scatter over RoCE for GB10, no NCCL.
//
// Same transport as arx_vllm.cu (pinned host buffers are GPU buffers at full
// speed on GB10, and the ConnectX DMAs them without GPUDirect; a CPU proxy
// posts RDMA writes when the GPU publishes a sequence number), sized for
// sequence-parallel prefill: tens of MB per rank per call.
//
// all_gather: the GPU copies this rank's slice into its slot of out[seq % 3];
// the proxy writes that slot into the same place in every peer's out[seq % 3],
// then a flag; the GPU waits for every peer's flags. The result is a view of
// out[seq % 3], valid until three more all-gathers have run.
// reduce_scatter: the GPU copies its full input into send[seq & 1]; the proxy
// writes chunk j into peer j's recv[seq & 1][rank], then a flag; the GPU waits
// and sums the world chunks for this rank in rank order (fp32).
//
// Every write is split in half, one half per ConnectX root, each followed by
// its own flag on that root's QP. Rank r sends to r+1, r+2, ... in turn.
// Reuse is safe for the same reason as in arx: a peer can only reach seq + 2
// (or + 3) after this rank posted seq + 1, which it does after its stream
// finished with seq.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define IBCK(x) do { if (!(x)) { fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #x, strerror(errno)); exit(1); } } while (0)

namespace {

constexpr int kMaxWorld = 8, kAgSlots = 3, kChunks = 8, kRing = 256;
// kRsPiece: one row range of one destination's chunk, published by a producer
// kernel as soon as it has written those rows (see finalize_rs_kernel).
enum Op : uint64_t { kAllGather = 1, kReduceScatter = 2, kRsPiece = 3 };

// Each op goes out as kChunks pieces so the network starts on piece 0 while
// the GPU still copies piece 1. A peer's flag for op seq, piece c, is
// seq * 16 + c + 1: monotonic, and it covers every earlier piece on that QP.
// Producers publish in any order, so each ring entry carries its own index + 1
// (tag), written last; the proxy consumes entries strictly in index order.
struct Work {
  uint64_t op, seq, chunk, slice, lo, hi;
  volatile uint64_t tag;
};
struct Ctl {                    // pinned; written by the GPU, read by the proxy
  Work ring[kRing];
  volatile uint64_t stop;
};

struct PeerInfo {
  uint32_t qpn[kMaxWorld][2];
  uint8_t gid[2][16];
  uint64_t ag_addr, rs_addr, flag_addr;
  uint32_t ag_rkey[2], rs_rkey[2], flag_rkey[2];
};

struct Dev {
  uint8_t* ag;                    // [kAgSlots][slot_bytes]
  uint8_t* rs_send;               // [2][slot_bytes]
  uint8_t* rs_recv;               // [2][world][slot_bytes / world]
  const volatile uint64_t* flag;  // [2 ops][world][2 roots]
  Ctl* ctl;
  unsigned* blocks;               // [kChunks * kMaxWorld] blocks done with each piece
  unsigned long long* work_next;  // next ring index to hand out
  unsigned long long* ready;      // highest flag_of(seq, piece) whose data has fully arrived
  int rank, world;
  size_t slot_bytes;
};

__device__ __forceinline__ uint64_t ld_acquire_sys(const volatile uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_cv(const void* p) {
  uint4 v;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

#ifndef RS_CACHED
#define RS_CACHED 1
#endif
__device__ __forceinline__ uint4 ld_plain(const void* p) { return *reinterpret_cast<const uint4*>(p); }
#if RS_CACHED
#define RS_LOAD ld_plain
#else
#define RS_LOAD ld_cv
#endif

__device__ __forceinline__ uint64_t flag_of(uint64_t seq, int c) { return seq * 16 + c + 1; }

// Piece c of one rank's slice: [lo, hi) bytes, 16-aligned.
__device__ __forceinline__ void piece(size_t slice, int c, size_t& lo, size_t& hi) {
  const size_t step = (slice / kChunks + 15) & ~size_t(15);
  lo = min(slice, step * c);
  hi = c == kChunks - 1 ? slice : min(slice, step * (c + 1));
}

__device__ void publish(const Dev& d, uint64_t op, uint64_t seq, uint64_t chunk, uint64_t slice, uint64_t lo,
                        uint64_t hi) {
  const unsigned long long idx = atomicAdd(d.work_next, 1ull);
  Work& it = d.ctl->ring[idx % kRing];
  it.op = op; it.seq = seq; it.chunk = chunk; it.slice = slice; it.lo = lo; it.hi = hi;
  __threadfence_system();
  it.tag = idx + 1;
}

// Copy piece c of every slice in `slices` (count of them, stride `slice`) from
// src to dst; the last block to finish a piece publishes it.
__device__ void copy_publish(const Dev& d, const uint8_t* src, uint8_t* dst, size_t slice, int slices, uint64_t op,
                             uint64_t seq) {
  for (int c = 0; c < kChunks; ++c) {
    size_t lo, hi;
    piece(slice, c, lo, hi);
    const size_t len = hi - lo, total = len * slices;
    const size_t stride = (size_t)gridDim.x * blockDim.x * 16;
    if (src != dst)
      for (size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16; i < total; i += stride) {
        const size_t s_ = i / len, o = s_ * slice + lo + i % len;
        *reinterpret_cast<uint4*>(dst + o) = *reinterpret_cast<const uint4*>(src + o);
      }
    __threadfence_system();
    __syncthreads();
    // One counter per piece: blocks do not wait for each other between pieces.
    if (threadIdx.x == 0 && atomicAdd(&d.blocks[c], 1) == gridDim.x - 1) {
      d.blocks[c] = 0;
      publish(d, op, seq, c, slice, lo, hi);
    }
  }
}

// Wait until every peer's flags (both roots) for op reach seq, piece c. Only
// block 0 reads the pinned flags the NIC writes; it relays progress through a
// device word, since many blocks polling those lines slows the NIC's writes.
__device__ void wait_piece(const Dev& d, uint64_t op, uint64_t seq, int c) {
  const uint64_t want = flag_of(seq, c);
  if (blockIdx.x == 0) {
    const volatile uint64_t* f = d.flag + (op == kAllGather ? 0 : d.world * 2);
    if (threadIdx.x < 2 * d.world && threadIdx.x / 2 != d.rank)
      while (ld_acquire_sys(f + threadIdx.x) < want) {}
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicMax(d.ready, (unsigned long long)want); }
  } else {
    if (threadIdx.x == 0)
      while (true) {
        unsigned long long g;
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(g) : "l"(d.ready) : "memory");
        if (g >= want) break;
      }
    __syncthreads();
  }
}

__global__ void ag_kernel(Dev d, const uint8_t* __restrict__ in, size_t slice, uint64_t seq) {
  uint8_t* out = d.ag + (seq % kAgSlots) * d.slot_bytes;
  copy_publish(d, in, out + d.rank * slice, slice, 1, kAllGather, seq);
  wait_piece(d, kAllGather, seq, kChunks - 1);
}

// out [slice / 2] bf16 = sum over ranks of their chunk `rank` of in.
__global__ void rs_kernel(Dev d, const uint8_t* __restrict__ in, size_t slice, __nv_bfloat16* __restrict__ out,
                          uint64_t seq) {
  const int par = seq & 1;
  uint8_t* send = d.rs_send + par * d.slot_bytes;
  copy_publish(d, in, send, slice, d.world, kReduceScatter, seq);
  const uint8_t* recv = d.rs_recv + (size_t)par * d.slot_bytes;
  const size_t stride = (size_t)gridDim.x * blockDim.x * 16;
  for (int c = 0; c < kChunks; ++c) {
    size_t lo, hi;
    piece(slice, c, lo, hi);
    wait_piece(d, kReduceScatter, seq, c);
    for (size_t i = lo + ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16; i < hi; i += stride) {
      float acc[8] = {};
      for (int src = 0; src < d.world; ++src) {  // rank order: every rank rounds the same way
        const uint4 v = src == d.rank ? *reinterpret_cast<const uint4*>(send + (size_t)d.rank * slice + i)
                                      : RS_LOAD(recv + (size_t)src * slice + i);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(b[k]);
      }
      __align__(16) __nv_bfloat16 o[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(acc[k]);
      *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(out) + i) = *reinterpret_cast<const uint4*>(o);
    }
  }
}


// Rows [lo, hi) of piece k of a destination's Rr-row chunk.
__device__ __host__ __forceinline__ void row_piece(int64_t Rr, int k, int64_t& lo, int64_t& hi) {
  lo = Rr * k / kChunks;
  hi = Rr * (k + 1) / kChunks;
}

// MoE output straight into the reduce-scatter send buffer:
//   row t = shared[t] + sum_k w[t, k] * y[pos[t, k]]  (fp32, rounded once),
// zero for padding rows t >= T. Blocks take rows piece-major (piece 0 of
// every destination first) and the last block of a piece publishes it, so
// the network starts while later rows are still being computed.
__global__ void finalize_rs_kernel(Dev d, const __nv_bfloat16* __restrict__ y, const int* __restrict__ pos,
                                   const float* __restrict__ w, const __nv_bfloat16* __restrict__ shared, int64_t T,
                                   int64_t Tpad, int H, int topk, uint64_t seq) {
  const int64_t Rr = Tpad / d.world;
  int64_t b = blockIdx.x, lo = 0, hi = 0;
  int k = 0;
  for (; k < kChunks; ++k) {
    row_piece(Rr, k, lo, hi);
    const int64_t n = (hi - lo) * d.world;
    if (b < n) break;
    b -= n;
  }
  const int j = (int)(b / (hi - lo));
  const int64_t t = j * Rr + lo + b % (hi - lo);
  __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(d.rs_send + (seq & 1) * d.slot_bytes) + t * H;
  __shared__ int ps[16];
  __shared__ float ws[16];
  if (t < T && threadIdx.x < topk) {
    ps[threadIdx.x] = pos[t * topk + threadIdx.x];
    ws[threadIdx.x] = w[t * topk + threadIdx.x];
  }
  __syncthreads();
  for (int c = threadIdx.x * 8; c < H; c += blockDim.x * 8) {
    float acc[8] = {};
    if (t < T) {
      const uint4 sv = *reinterpret_cast<const uint4*>(shared + t * H + c);
      const __nv_bfloat162* sb = reinterpret_cast<const __nv_bfloat162*>(&sv);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(sb[i]);
        acc[2 * i] = f.x; acc[2 * i + 1] = f.y;
      }
      for (int q = 0; q < topk; ++q) {
        const uint4 v = *reinterpret_cast<const uint4*>(y + (size_t)ps[q] * H + c);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 f = __bfloat1622float2(b2[i]);
          acc[2 * i] += ws[q] * f.x;
          acc[2 * i + 1] += ws[q] * f.y;
        }
      }
    }
    uint4 o;
    __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) ob[i] = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
    *reinterpret_cast<uint4*>(dst + c) = o;
  }
  __threadfence_system();
  __syncthreads();
  const int idx = k * d.world + j;
  if (threadIdx.x == 0 && atomicAdd(&d.blocks[idx], 1) == (unsigned)(hi - lo) - 1) {
    d.blocks[idx] = 0;
    const uint64_t row = (uint64_t)H * 2;
    publish(d, kRsPiece, seq, (uint64_t)j * kChunks + k, Rr * row, lo * row, hi * row);
  }
}

// out [Rr, H] = sum over ranks, in rank order, of their rows for this rank,
// piece by piece as the pieces arrive.
__global__ void rs_finish_kernel(Dev d, __nv_bfloat16* __restrict__ out, int64_t Rr, int H, uint64_t seq) {
  const int par = seq & 1;
  const size_t row = (size_t)H * 2, slice = Rr * row;
  const uint8_t* send = d.rs_send + par * d.slot_bytes + d.rank * slice;
  const uint8_t* recv = d.rs_recv + (size_t)par * d.slot_bytes;
  const size_t stride = (size_t)gridDim.x * blockDim.x * 16;
  for (int k = 0; k < kChunks; ++k) {
    int64_t lo, hi;
    row_piece(Rr, k, lo, hi);
    wait_piece(d, kReduceScatter, seq, k);
    for (size_t i = lo * row + ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16; i < hi * row; i += stride) {
      float acc[8] = {};
      for (int src = 0; src < d.world; ++src) {
        const uint4 v = src == d.rank ? *reinterpret_cast<const uint4*>(send + i)
                                      : RS_LOAD(recv + (size_t)src * slice + i);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[q] += __bfloat162float(b[q]);
      }
      __align__(16) __nv_bfloat16 o[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) o[q] = __float2bfloat16(acc[q]);
      *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(out) + i) = *reinterpret_cast<const uint4*>(o);
    }
  }
}

struct State {
  int rank = -1, world = 0, gid_idx = 0;
  size_t slot_bytes = 0, rs_slot_bytes = 0;
  uint8_t *ag = nullptr, *rs_send = nullptr, *rs_recv = nullptr;
  uint64_t* flag = nullptr;
  Ctl* ctl = nullptr;
  ibv_context* ctx[2] = {};
  ibv_pd* pd[2] = {};
  ibv_mr *ag_mr[2] = {}, *rs_send_mr[2] = {}, *rs_recv_mr[2] = {}, *flag_mr[2] = {};
  ibv_cq* cq[2] = {};
  ibv_qp* qp[kMaxWorld][2] = {};
  PeerInfo mine{};
  std::vector<PeerInfo> all;
  Dev dev{};
  bool connected = false;
};
State S;
std::atomic<bool> g_err{false};

bool post(int j, int r, uint64_t laddr, uint32_t lkey, uint64_t len, uint64_t raddr, uint32_t rkey, uint64_t flag_off,
          uint64_t* seq_word) {
  ibv_sge sg{laddr, (uint32_t)len, lkey};
  ibv_sge fs{(uint64_t)seq_word, 8, 0};
  ibv_send_wr w{}, f{}, *bad;
  w.opcode = IBV_WR_RDMA_WRITE; w.sg_list = &sg; w.num_sge = 1;
  w.wr.rdma.remote_addr = raddr; w.wr.rdma.rkey = rkey;
  f.opcode = IBV_WR_RDMA_WRITE; f.sg_list = &fs; f.num_sge = 1;
  f.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  f.wr_id = (uint64_t)j;
  f.wr.rdma.remote_addr = S.all[j].flag_addr + flag_off; f.wr.rdma.rkey = S.all[j].flag_rkey[r];
  w.next = len ? &f : nullptr;
  ibv_send_wr* head = len ? &w : &f;
  if (int e = ibv_post_send(S.qp[j][r], head, &bad)) {
    fprintf(stderr, "arxbig rank %d: post to %d root %d failed: %s\n", S.rank, j, r, strerror(e));
    return false;
  }
  return true;
}

// Sends one piece [lo, hi) of op `it` to peer j, split over both roots, each
// half followed by a flag write of `flag` on that root.
bool send_piece(const Work& it, int j, uint64_t* flag) {
  const int rank = S.rank, world = S.world;
  const uint64_t len = it.hi - it.lo, half = (len / 2 + 15) & ~15ull;
  for (int r = 0; r < 2; ++r) {
    const uint64_t off = it.lo + (r ? half : 0), n = r ? len - half : half;
    bool ok;
    if (it.op == kAllGather) {
      const uint64_t at = (it.seq % kAgSlots) * S.slot_bytes + rank * it.slice + off;
      ok = post(j, r, (uint64_t)S.ag + at, S.ag_mr[r]->lkey, n, S.all[j].ag_addr + at, S.all[j].ag_rkey[r],
                (rank * 2 + r) * 8, flag);
    } else {
      const uint64_t par = it.seq & 1;
      ok = post(j, r, (uint64_t)S.rs_send + par * S.rs_slot_bytes + j * it.slice + off, S.rs_send_mr[r]->lkey, n,
                S.all[j].rs_addr + par * S.rs_slot_bytes + rank * it.slice + off, S.all[j].rs_rkey[r],
                (world * 2 + rank * 2 + r) * 8, flag);
    }
    if (!ok) return false;
  }
  return true;
}

void proxy_loop() {
  uint64_t done = 0;
  static Work pend[kMaxWorld][kChunks];  // kRsPiece items that arrived early
  uint32_t pend_mask[kMaxWorld] = {}, next_k[kMaxWorld] = {};
  uint64_t cur_seq[kMaxWorld] = {};
  const int rank = S.rank, world = S.world;
  while (!S.ctl->stop) {
    for (int r = 0; r < 2; ++r) {
      ibv_wc wc[32];
      const int n = ibv_poll_cq(S.cq[r], 32, wc);
      for (int i = 0; i < n; ++i)
        if (wc[i].status != IBV_WC_SUCCESS) {
          fprintf(stderr, "arxbig rank %d: completion error %d\n", rank, wc[i].status);
          g_err = true;
          return;
        }
    }
    Work& slot = S.ctl->ring[done % kRing];
    if (slot.tag != done + 1) continue;
    std::atomic_thread_fence(std::memory_order_acquire);
    Work it;
    it.op = slot.op; it.seq = slot.seq; it.chunk = slot.chunk; it.slice = slot.slice; it.lo = slot.lo; it.hi = slot.hi;
    ++done;
    if (it.op != kRsPiece) {  // one piece for every peer
      uint64_t flag = it.seq * 16 + it.chunk + 1;  // inline: copied at post time
      for (int k = 1; k < world; ++k)
        if (!send_piece(it, (rank + k) % world, &flag)) { g_err = true; return; }
      continue;
    }
    // Row piece `sub` of destination j's chunk. Peers read the flag as "all
    // pieces up to sub arrived", so pieces go out in order per destination.
    const int j = (int)(it.chunk / kChunks), sub = (int)(it.chunk % kChunks);
    if (j == rank) continue;  // this rank's own rows never leave
    if (cur_seq[j] != it.seq) { cur_seq[j] = it.seq; next_k[j] = 0; pend_mask[j] = 0; }
    pend[j][sub] = it;
    pend_mask[j] |= 1u << sub;
    while (next_k[j] < kChunks && (pend_mask[j] >> next_k[j] & 1)) {
      const int k = next_k[j]++;
      uint64_t flag = it.seq * 16 + k + 1;
      if (!send_piece(pend[j][k], j, &flag)) { g_err = true; return; }
    }
  }
}

Dev make_dev() {
  Dev d{};
  d.ag = S.ag; d.rs_send = S.rs_send; d.rs_recv = S.rs_recv; d.flag = S.flag; d.ctl = S.ctl;
  CK(cudaMalloc(&d.blocks, kChunks * kMaxWorld * 4)); CK(cudaMalloc(&d.ready, 8)); CK(cudaMalloc(&d.work_next, 8));
  CK(cudaMemset(d.blocks, 0, kChunks * kMaxWorld * 4)); CK(cudaMemset(d.ready, 0, 8)); CK(cudaMemset(d.work_next, 0, 8));
  d.rank = S.rank; d.world = S.world; d.slot_bytes = S.slot_bytes;
  return d;
}

}  // namespace

// rs_slot_bytes: 0 leaves reduce_scatter unavailable and saves its buffers.
py::bytes prepare(int64_t rank, int64_t world, std::string dev0, std::string dev1, int64_t gid_idx, int64_t slot_bytes,
                  int64_t rs_slot_bytes) {
  TORCH_CHECK(S.rank < 0 && world >= 2 && world <= kMaxWorld && slot_bytes % (world * 16) == 0);
  TORCH_CHECK(rs_slot_bytes == 0 || rs_slot_bytes == slot_bytes, "arxbig: reduce_scatter uses the same slot size");
  S.rank = rank; S.world = world; S.gid_idx = gid_idx; S.slot_bytes = slot_bytes; S.rs_slot_bytes = rs_slot_bytes;
  const size_t rs_alloc = rs_slot_bytes ? 2 * rs_slot_bytes : 4096;
  CK(cudaHostAlloc(&S.ag, kAgSlots * slot_bytes, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.rs_send, rs_alloc, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.rs_recv, rs_alloc, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.flag, 4 * kMaxWorld * sizeof(uint64_t), cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.ctl, sizeof(Ctl), cudaHostAllocMapped));
  memset(S.flag, 0, 4 * kMaxWorld * sizeof(uint64_t));
  memset((void*)S.ctl, 0, sizeof(Ctl));
  const char* names[2] = {dev0.c_str(), dev1.c_str()};
  int ndev;
  ibv_device** list = ibv_get_device_list(&ndev);
  const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  for (int r = 0; r < 2; ++r) {
    for (int i = 0; i < ndev; ++i)
      if (!strcmp(ibv_get_device_name(list[i]), names[r])) S.ctx[r] = ibv_open_device(list[i]);
    TORCH_CHECK(S.ctx[r], "arxbig: no RDMA device ", names[r]);
    S.pd[r] = ibv_alloc_pd(S.ctx[r]); IBCK(S.pd[r]);
    S.ag_mr[r] = ibv_reg_mr(S.pd[r], S.ag, kAgSlots * slot_bytes, acc); IBCK(S.ag_mr[r]);
    S.rs_send_mr[r] = ibv_reg_mr(S.pd[r], S.rs_send, rs_alloc, acc); IBCK(S.rs_send_mr[r]);
    S.rs_recv_mr[r] = ibv_reg_mr(S.pd[r], S.rs_recv, rs_alloc, acc); IBCK(S.rs_recv_mr[r]);
    S.flag_mr[r] = ibv_reg_mr(S.pd[r], S.flag, 4 * kMaxWorld * sizeof(uint64_t), acc); IBCK(S.flag_mr[r]);
    S.cq[r] = ibv_create_cq(S.ctx[r], 4096, nullptr, nullptr, 0); IBCK(S.cq[r]);
    ibv_gid gid;
    IBCK(ibv_query_gid(S.ctx[r], 1, gid_idx, &gid) == 0);
    memcpy(S.mine.gid[r], gid.raw, 16);
    S.mine.ag_rkey[r] = S.ag_mr[r]->rkey;
    S.mine.rs_rkey[r] = S.rs_recv_mr[r]->rkey;
    S.mine.flag_rkey[r] = S.flag_mr[r]->rkey;
  }
  ibv_free_device_list(list);
  for (int j = 0; j < world; ++j)
    for (int r = 0; r < 2; ++r) {
      if (j == rank) continue;
      ibv_qp_init_attr ia{};
      ia.send_cq = S.cq[r]; ia.recv_cq = S.cq[r]; ia.qp_type = IBV_QPT_RC;
      ia.cap.max_send_wr = 1024; ia.cap.max_recv_wr = 1; ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
      ia.cap.max_inline_data = 64;
      S.qp[j][r] = ibv_create_qp(S.pd[r], &ia); IBCK(S.qp[j][r]);
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = 1; a.qp_access_flags = acc;
      IBCK(ibv_modify_qp(S.qp[j][r], &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0);
      S.mine.qpn[j][r] = S.qp[j][r]->qp_num;
    }
  S.mine.ag_addr = (uint64_t)S.ag;
  S.mine.rs_addr = (uint64_t)S.rs_recv;
  S.mine.flag_addr = (uint64_t)S.flag;
  return py::bytes(reinterpret_cast<const char*>(&S.mine), sizeof(PeerInfo));
}

void connect(std::vector<std::string> infos) {
  TORCH_CHECK(S.rank >= 0 && !S.connected && (int)infos.size() == S.world);
  S.all.resize(S.world);
  for (int j = 0; j < S.world; ++j) {
    TORCH_CHECK(infos[j].size() == sizeof(PeerInfo), "arxbig: PeerInfo size mismatch");
    memcpy(&S.all[j], infos[j].data(), sizeof(PeerInfo));
  }
  for (int j = 0; j < S.world; ++j)
    for (int r = 0; r < 2; ++r) {
      if (j == S.rank) continue;
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_RTR; a.path_mtu = IBV_MTU_4096; a.dest_qp_num = S.all[j].qpn[S.rank][r]; a.rq_psn = 0;
      a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
      a.ah_attr.is_global = 1; a.ah_attr.port_num = 1; a.ah_attr.grh.hop_limit = 1;
      a.ah_attr.grh.sgid_index = S.gid_idx;
      memcpy(a.ah_attr.grh.dgid.raw, S.all[j].gid[r], 16);
      IBCK(ibv_modify_qp(S.qp[j][r], &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0);
      a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = 0; a.max_rd_atomic = 1;
      IBCK(ibv_modify_qp(S.qp[j][r], &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                              IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0);
    }
  S.dev = make_dev();
  CK(cudaDeviceSynchronize());
  std::thread(proxy_loop).detach();
  S.connected = true;
}

static int blocks_for(size_t bytes) { return (int)std::max<size_t>(1, std::min<size_t>(48, bytes / (256 * 16 * 8))); }
static uint64_t g_seq = 0;  // collectives run in stream order, so the host numbers them

// in: this rank's slice (any dtype, contiguous). Returns the gathered data as a
// view of the pinned output slot, shaped [world * in.size(0), ...]; it stays
// valid until three more all-gathers have run.
torch::Tensor all_gather(torch::Tensor in) {
  TORCH_CHECK(S.connected && !g_err, "arxbig is not connected or its proxy failed");
  TORCH_CHECK(in.is_contiguous() && in.is_cuda() && in.dim() >= 1);
  const size_t slice = in.numel() * in.element_size();
  TORCH_CHECK(slice % 16 == 0 && slice * S.world <= S.slot_bytes, "arxbig: all_gather slice size");
  const uint64_t seq = ++g_seq;
  ag_kernel<<<blocks_for(slice), 256, 0, at::cuda::getCurrentCUDAStream()>>>(S.dev, (const uint8_t*)in.data_ptr(),
                                                                              slice, seq);
  std::vector<int64_t> shape(in.sizes().begin(), in.sizes().end());
  shape[0] *= S.world;
  return torch::from_blob(S.ag + (seq % kAgSlots) * S.slot_bytes, shape, in.options());
}

// in: bf16 [world * m, ...] contiguous. Returns bf16 [m, ...]: the sum over
// ranks of each rank's chunk `rank`.
torch::Tensor reduce_scatter(torch::Tensor in) {
  TORCH_CHECK(S.connected && !g_err, "arxbig is not connected or its proxy failed");
  TORCH_CHECK(S.rs_slot_bytes, "arxbig: prepared without reduce_scatter buffers");
  TORCH_CHECK(in.is_contiguous() && in.is_cuda() && in.scalar_type() == at::kBFloat16 && in.size(0) % S.world == 0);
  const size_t slice = in.numel() * 2 / S.world;
  TORCH_CHECK(slice % 16 == 0 && slice * S.world <= S.slot_bytes, "arxbig: reduce_scatter size");
  std::vector<int64_t> shape(in.sizes().begin(), in.sizes().end());
  shape[0] /= S.world;
  auto out = torch::empty(shape, in.options());
  const uint64_t seq = ++g_seq;
  rs_kernel<<<blocks_for(slice * S.world), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.dev, (const uint8_t*)in.data_ptr(), slice, (__nv_bfloat16*)out.data_ptr(), seq);
  return out;
}

// The pinned buffer the next reduce_scatter sends from, as a bf16 tensor of
// `shape`. A producer that writes its output here saves reduce_scatter's copy;
// it stays valid until that reduce_scatter.
torch::Tensor rs_input(std::vector<int64_t> shape) {
  TORCH_CHECK(S.connected && S.rs_slot_bytes);
  int64_t n = 1;
  for (auto v : shape) n *= v;
  TORCH_CHECK(n * 2 <= (int64_t)S.slot_bytes, "arxbig: rs_input too large");
  const uint64_t par = (g_seq + 1) & 1;
  return torch::from_blob(S.rs_send + par * S.slot_bytes, shape,
                          torch::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA));
}

// The MoE's final combine, reduce-scattered: finalize_rs_kernel writes and
// publishes rows; rs_finish (called next on the same stream) returns this
// rank's [Tpad / world, H] sum. y bf16 [R, H], pos int32 [T * topk],
// w fp32 [T, topk] (routing weights, scale folded in), shared bf16 [T, H].
int64_t moe_finalize_rs(torch::Tensor y, torch::Tensor pos, torch::Tensor w, torch::Tensor shared, int64_t T,
                        int64_t Tpad) {
  TORCH_CHECK(S.connected && !g_err && S.rs_slot_bytes, "arxbig: reduce_scatter unavailable");
  const int H = y.size(1), topk = w.size(1);
  TORCH_CHECK(Tpad % S.world == 0 && Tpad / S.world >= kChunks && T <= Tpad && topk <= 16 && H % 8 == 0);
  TORCH_CHECK((size_t)Tpad * H * 2 <= S.slot_bytes, "arxbig: MoE output larger than a slot");
  TORCH_CHECK(y.is_contiguous() && pos.is_contiguous() && w.is_contiguous() && shared.is_contiguous());
  const uint64_t seq = ++g_seq;
  finalize_rs_kernel<<<Tpad, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.dev, (const __nv_bfloat16*)y.data_ptr(), pos.data_ptr<int>(), w.data_ptr<float>(),
      (const __nv_bfloat16*)shared.data_ptr(), T, Tpad, H, topk, seq);
  return (int64_t)seq;
}

torch::Tensor rs_finish(int64_t seq, int64_t rows, int64_t H) {
  TORCH_CHECK((uint64_t)seq == g_seq, "arxbig: rs_finish must follow its producer directly");
  auto out = torch::empty({rows, H}, torch::TensorOptions().dtype(at::kBFloat16).device(torch::kCUDA));
  rs_finish_kernel<<<blocks_for(rows * H * 2 * S.world), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.dev, (__nv_bfloat16*)out.data_ptr(), rows, (int)H, (uint64_t)seq);
  return out;
}

bool failed() { return g_err; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &prepare);
  m.def("connect", &connect);
  m.def("all_gather", &all_gather);
  m.def("reduce_scatter", &reduce_scatter);
  m.def("rs_input", &rs_input);
  m.def("moe_finalize_rs", &moe_finalize_rs);
  m.def("rs_finish", &rs_finish);
  m.def("failed", &failed);
}
