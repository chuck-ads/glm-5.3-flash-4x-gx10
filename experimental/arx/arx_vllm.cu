// arx: one-shot all-reduce over RoCE for GB10, no NCCL.
//
// GB10's CPU and GPU share one coherent memory, so pinned host buffers are
// GPU buffers at full speed and the ConnectX can DMA into them without
// GPUDirect. Per all-reduce:
//   1. the GPU writes its partial into send[seq & 1] and publishes seq;
//   2. a CPU proxy sees seq, RDMA-writes the partial into every peer's
//      recv[seq & 1][rank], then writes seq into the peer's flag[rank]
//      (same QP, so the data is placed first);
//   3. the GPU waits for every peer's flag to reach seq and sums.
// Two buffers by parity are enough: a rank reaches seq + 2 only after every
// peer consumed seq + 1, which needed this rank's seq + 1 flag, which the
// proxy posts after this rank's seq data was read.
//
// The proxy posts every seq in order. The GPU can publish seq + 1 before the
// proxy has seen seq (finishing seq needs only the peers' data), and skipping
// seq would let peers accept seq + 1's flag over stale seq data.
//
// Every partial is split in half, one half per ConnectX root, each followed by
// its own flag on that root's QP: flag[src * 2 + root]. Rank r sends to
// r+1, r+2, ... in turn, so no receiver takes every sender at once.
//
// vLLM form: prepare() returns this rank's connection details, the caller
// all-gathers them over its own group, connect() wires the QPs and starts the
// proxy, then allreduce(in, out) runs on the current stream and is
// CUDA-graph capturable. in/out are bf16 device tensors.
#include <arpa/inet.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define IBCK(x) do { if (!(x)) { fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__, #x, strerror(errno)); exit(1); } } while (0)

constexpr int kMaxWorld = 8;
constexpr size_t kMaxBytes = 512 << 10;  // one partial: bf16 [32, 4096] is 256 KB

struct Ctl {                   // pinned; written by the GPU, read by the proxy
  volatile uint64_t seq;       // last published partial
  volatile uint64_t bytes[2];  // size of the partial in send[parity]
  volatile uint64_t stop;
  volatile uint64_t done;      // last seq the GPU finished summing (for the watchdog)
};

struct PeerInfo {  // what rank r tells everyone about its QP towards peer j
  uint32_t qpn[kMaxWorld][2];  // towards peer j, per root
  uint8_t gid[2][16];          // per ConnectX root
  uint64_t recv_addr, flag_addr;
  uint32_t recv_rkey[2], flag_rkey[2];
};

// ---- GPU side ---------------------------------------------------------------

__device__ __forceinline__ uint64_t ld_acquire_sys(const volatile uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint4 ld_cv(const void* p) {  // bypass caches: the NIC wrote it
  uint4 v;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}


struct Dev {
  __nv_bfloat16* send;           // [2][kMaxBytes / 2]
  const __nv_bfloat16* recv;     // [2][world][kMaxBytes / 2]
  const volatile uint64_t* flag; // [2 * world]
  Ctl* ctl;
  unsigned* seq_dev;             // device: last completed seq
  unsigned* blocks;         // device [2]: blocks past the copy, blocks finished
  unsigned* go;                  // device: seq whose peer data has all arrived
  int rank, world;
};

// Weight ranges the next kernels read; the kernel asks L2 for them before it waits.
constexpr int kMaxPrefetch = 4;
struct Prefetch {
  const uint8_t* ptr[kMaxPrefetch];
  long long bytes[kMaxPrefetch];
  int count;
};

// out = sum over ranks of in. in and out may alias.
__global__ void arx_kernel(Dev d, const __nv_bfloat16* __restrict__ in, __nv_bfloat16* out, int n, Prefetch pf) {
  const uint64_t seq = *d.seq_dev + 1;
  const int par = seq & 1;
  __nv_bfloat16* mine = d.send + (size_t)par * (kMaxBytes / 2);
  const int stride = gridDim.x * blockDim.x * 8;
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < n; i += stride)
    *reinterpret_cast<uint4*>(mine + i) = *reinterpret_cast<const uint4*>(in + i);
  __threadfence_system();
  __syncthreads();
  // Separate counters per phase: blocks that start late would otherwise add
  // their first count after early blocks added their second, and the
  // publish would be skipped or would go out before every block copied.
  if (threadIdx.x == 0 && atomicAdd(&d.blocks[0], 1) == gridDim.x - 1) {  // last block publishes
    d.blocks[0] = 0;
    d.ctl->bytes[par] = (uint64_t)n * 2;
    __threadfence_system();
    d.ctl->seq = seq;
  }
  // DRAM is idle while the peers' data is in flight: spread one bulk L2
  // prefetch per thread over each range.
  if (pf.count > 0) {
    const long long nthr = (long long)gridDim.x * blockDim.x, t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = 0; r < pf.count; ++r) {
      const long long chunk = ((pf.bytes[r] + nthr - 1) / nthr + 15) & ~15LL, off = t * chunk;
      const long long len = min(chunk, pf.bytes[r] - off) & ~15LL;
      if (len > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(pf.ptr[r] + off), "r"((unsigned)len));
    }
  }
  // Block 0 polls the pinned flags the NIC writes; the others wait on a word in
  // device memory, so only a few threads touch lines the NIC is filling.
  if (blockIdx.x == 0) {
    if (threadIdx.x < 2 * d.world && threadIdx.x / 2 != d.rank)
      while (ld_acquire_sys(d.flag + threadIdx.x) < seq) {}
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence(); atomicExch(d.go, (unsigned)seq); }
  } else {
    if (threadIdx.x == 0)
      while (true) {
        unsigned g;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(g) : "l"(d.go) : "memory");
        if (g >= (unsigned)seq) break;
      }
    __syncthreads();
  }
  for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8; i < n; i += stride) {
    uint4 v[kMaxWorld];
#pragma unroll
    for (int src = 0; src < kMaxWorld; ++src)
      if (src < d.world)
        v[src] = src == d.rank ? *reinterpret_cast<const uint4*>(mine + i)
                               : ld_cv(d.recv + ((size_t)par * d.world + src) * (kMaxBytes / 2) + i);
    float acc[8] = {};
#pragma unroll
    for (int src = 0; src < kMaxWorld; ++src)  // rank order: every rank gets the same bits
      if (src < d.world) {
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v[src]);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(b[k]);
      }
    __align__(16) __nv_bfloat16 o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
  }
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&d.blocks[1], 1) == gridDim.x - 1) {  // last block out
    d.blocks[1] = 0;
    *d.seq_dev = (unsigned)seq;
    d.ctl->done = seq;
  }
}

// ---- host side ---------------------------------------------------------------

namespace {
struct State {
  int rank = -1, world = 0, gid_idx = 0;
  uint8_t *send_h = nullptr, *recv_h = nullptr;
  uint64_t* flag_h = nullptr;
  Ctl* ctl = nullptr;
  ibv_context* ctx[2] = {};
  ibv_pd* pd[2] = {};
  ibv_mr *send_mr[2] = {}, *recv_mr[2] = {}, *flag_mr[2] = {};
  ibv_cq* cq[2] = {};
  ibv_qp* qp[kMaxWorld][2] = {};
  PeerInfo mine{};
  std::vector<PeerInfo> all;
  Dev dev{};
  bool connected = false;
};
State S;
std::atomic<bool> g_proxy_err{false};

void proxy_loop() {
  uint64_t seq = 0, posted[kMaxWorld][2] = {};
  const int rank = S.rank, world = S.world;
  auto last_move = std::chrono::steady_clock::now();
  uint64_t last_done = 0;
  bool reported = false;
  while (!S.ctl->stop) {
    if (S.ctl->seq <= seq) {
      // Watchdog: published work that has not completed for 2 s is a stall,
      // and so is a peer whose flags are ahead of what this rank posted.
      // Print what this rank sent and what it has heard from each peer.
      const uint64_t done = S.ctl->done;
      const auto now = std::chrono::steady_clock::now();
      bool behind = false;
      for (int j = 0; j < world; ++j)
        if (j != rank) behind |= S.flag_h[j * 2] > seq;
      if (done != last_done || (done >= seq && !behind)) {
        last_done = done; last_move = now; reported = false;
      }
      else if (!reported && now - last_move > std::chrono::seconds(2)) {
        reported = true;
        char buf[512];
        int o = snprintf(buf, sizeof buf, "arx rank %d STALL: published %lu posted %lu done %lu; flags from peers:",
                         rank, (unsigned long)S.ctl->seq, (unsigned long)seq, (unsigned long)done);
        for (int j = 0; j < world; ++j)
          if (j != rank)
            o += snprintf(buf + o, sizeof buf - o, " r%d=%lu/%lu", j, (unsigned long)S.flag_h[j * 2],
                          (unsigned long)S.flag_h[j * 2 + 1]);
        fprintf(stderr, "%s\n", buf);
      }
      continue;
    }
    ++seq;  // every seq, in order
    std::atomic_thread_fence(std::memory_order_acquire);
    const int par = seq & 1;
    const uint32_t bytes = (uint32_t)S.ctl->bytes[par];
    const uint32_t half = (bytes / 2 + 15) & ~15u;  // root 0 takes the first half
    for (int step = 1; step < world; ++step) {
      const int j = (rank + step) % world;
      for (int r = 0; r < 2; ++r) {
        const uint32_t off = r ? half : 0, len = r ? bytes - half : half;
        ibv_sge sg{(uint64_t)(S.send_h + par * kMaxBytes + off), len, S.send_mr[r]->lkey};
        ibv_send_wr w{}, f{}, *bad;
        w.opcode = IBV_WR_RDMA_WRITE; w.sg_list = &sg; w.num_sge = 1;
        w.wr.rdma.remote_addr = S.all[j].recv_addr + ((uint64_t)par * world + rank) * kMaxBytes + off;
        w.wr.rdma.rkey = S.all[j].recv_rkey[r];
        uint64_t sv = seq;
        ibv_sge fs{(uint64_t)&sv, 8, 0};
        f.opcode = IBV_WR_RDMA_WRITE; f.sg_list = &fs; f.num_sge = 1; f.send_flags = IBV_SEND_INLINE;
        f.wr.rdma.remote_addr = S.all[j].flag_addr + (rank * 2 + r) * 8; f.wr.rdma.rkey = S.all[j].flag_rkey[r];
        // Unsignalled WRs are reclaimed only by a later signalled one on the
        // same QP, so the count is per QP.
        if ((++posted[j][r] & 31) == 0) f.send_flags |= IBV_SEND_SIGNALED;
        w.next = &f;
        if (int e = ibv_post_send(S.qp[j][r], &w, &bad)) {
          fprintf(stderr, "arx rank %d: post to %d root %d failed: %s\n", rank, j, r, strerror(e));
          g_proxy_err = true;
          return;
        }
      }
    }
    for (int r = 0; r < 2; ++r) {
      ibv_wc wc[16];
      const int n = ibv_poll_cq(S.cq[r], 16, wc);
      for (int i = 0; i < n; ++i)
        if (wc[i].status != IBV_WC_SUCCESS) {
          fprintf(stderr, "arx rank %d: completion error %d\n", rank, wc[i].status);
          g_proxy_err = true;
          return;
        }
    }
  }
}
}  // namespace

// Allocates buffers, opens both roots, creates the QPs; returns this rank's
// PeerInfo for the caller to all-gather.
py::bytes arx_prepare(int64_t rank, int64_t world, std::string dev0, std::string dev1, int64_t gid_idx) {
  TORCH_CHECK(S.rank < 0, "arx is already prepared in this process");
  TORCH_CHECK(world >= 2 && world <= kMaxWorld);
  S.rank = rank; S.world = world; S.gid_idx = gid_idx;
  CK(cudaHostAlloc(&S.send_h, 2 * kMaxBytes, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.recv_h, 2 * world * kMaxBytes, cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.flag_h, 2 * kMaxWorld * sizeof(uint64_t), cudaHostAllocMapped));
  CK(cudaHostAlloc(&S.ctl, sizeof(Ctl), cudaHostAllocMapped));
  memset(S.flag_h, 0, 2 * kMaxWorld * sizeof(uint64_t));
  memset((void*)S.ctl, 0, sizeof(Ctl));
  const char* names[2] = {dev0.c_str(), dev1.c_str()};
  int ndev;
  ibv_device** list = ibv_get_device_list(&ndev);
  const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  for (int r = 0; r < 2; ++r) {
    for (int i = 0; i < ndev; ++i)
      if (!strcmp(ibv_get_device_name(list[i]), names[r])) S.ctx[r] = ibv_open_device(list[i]);
    TORCH_CHECK(S.ctx[r], "arx: no RDMA device ", names[r]);
    S.pd[r] = ibv_alloc_pd(S.ctx[r]); IBCK(S.pd[r]);
    S.send_mr[r] = ibv_reg_mr(S.pd[r], S.send_h, 2 * kMaxBytes, acc); IBCK(S.send_mr[r]);
    S.recv_mr[r] = ibv_reg_mr(S.pd[r], S.recv_h, 2 * world * kMaxBytes, acc); IBCK(S.recv_mr[r]);
    S.flag_mr[r] = ibv_reg_mr(S.pd[r], S.flag_h, 2 * kMaxWorld * sizeof(uint64_t), acc); IBCK(S.flag_mr[r]);
    S.cq[r] = ibv_create_cq(S.ctx[r], 4096, nullptr, nullptr, 0); IBCK(S.cq[r]);
    ibv_gid gid;
    IBCK(ibv_query_gid(S.ctx[r], 1, gid_idx, &gid) == 0);
    memcpy(S.mine.gid[r], gid.raw, 16);
    S.mine.recv_rkey[r] = S.recv_mr[r]->rkey;
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
  S.mine.recv_addr = (uint64_t)S.recv_h;
  S.mine.flag_addr = (uint64_t)S.flag_h;
  return py::bytes(reinterpret_cast<const char*>(&S.mine), sizeof(PeerInfo));
}

// Takes every rank's PeerInfo in rank order, brings the QPs up, starts the
// proxy. The caller must barrier its group before the first allreduce.
void arx_connect(std::vector<std::string> infos) {
  TORCH_CHECK(S.rank >= 0 && !S.connected && (int)infos.size() == S.world);
  S.all.resize(S.world);
  for (int j = 0; j < S.world; ++j) {
    TORCH_CHECK(infos[j].size() == sizeof(PeerInfo), "arx: PeerInfo size mismatch");
    memcpy(&S.all[j], infos[j].data(), sizeof(PeerInfo));
  }
  for (int j = 0; j < S.world; ++j)
    for (int r = 0; r < 2; ++r) {
      if (j == S.rank) continue;
      ibv_qp_attr a{};
      ibv_port_attr port_attr{};
      IBCK(ibv_query_port(S.ctx[r], 1, &port_attr) == 0);
      a.qp_state = IBV_QPS_RTR; a.path_mtu = port_attr.active_mtu; a.dest_qp_num = S.all[j].qpn[S.rank][r]; a.rq_psn = 0;
      a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
      a.ah_attr.is_global = 1; a.ah_attr.port_num = 1; a.ah_attr.grh.hop_limit = 1;
      a.ah_attr.grh.sgid_index = S.gid_idx;
      memcpy(a.ah_attr.grh.dgid.raw, S.all[j].gid[r], 16);
      IBCK(ibv_modify_qp(S.qp[j][r], &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                             IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0);
      a = {};
      a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = 0; a.max_rd_atomic = 1;
      IBCK(ibv_modify_qp(S.qp[j][r], &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                             IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0);
    }
  Dev& d = S.dev;
  d.send = (__nv_bfloat16*)S.send_h; d.recv = (const __nv_bfloat16*)S.recv_h; d.flag = S.flag_h; d.ctl = S.ctl;
  d.rank = S.rank; d.world = S.world;
  CK(cudaMalloc(&d.seq_dev, 4)); CK(cudaMalloc(&d.blocks, 8)); CK(cudaMalloc(&d.go, 4));
  CK(cudaMemset(d.seq_dev, 0, 4)); CK(cudaMemset(d.blocks, 0, 8)); CK(cudaMemset(d.go, 0, 4));
  CK(cudaDeviceSynchronize());
  std::thread(proxy_loop).detach();
  S.connected = true;
}

void arx_allreduce(torch::Tensor in, torch::Tensor out, std::vector<int64_t> pf_ptrs, std::vector<int64_t> pf_bytes) {
  TORCH_CHECK(S.connected, "arx is not connected");
  TORCH_CHECK(!g_proxy_err, "arx proxy failed");
  TORCH_CHECK(in.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 && in.is_contiguous() &&
              out.is_contiguous() && in.numel() == out.numel());
  const int n = in.numel();
  TORCH_CHECK(n % 8 == 0 && (size_t)n * 2 <= kMaxBytes);
  const int threads = 256, blocks = std::max(1, std::min(8, n / (threads * 8)));
  Prefetch pf{};
  pf.count = (int)std::min<size_t>(pf_ptrs.size(), kMaxPrefetch);
  for (int r = 0; r < pf.count; ++r) {
    pf.ptr[r] = reinterpret_cast<const uint8_t*>(pf_ptrs[r]);
    pf.bytes[r] = pf_bytes[r];
  }
  arx_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      S.dev, (const __nv_bfloat16*)in.data_ptr(), (__nv_bfloat16*)out.data_ptr(), n, pf);
}

int64_t arx_max_bytes() { return kMaxBytes; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prepare", &arx_prepare);
  m.def("connect", &arx_connect);
  m.def("allreduce", &arx_allreduce);
  m.def("max_bytes", &arx_max_bytes);
}
