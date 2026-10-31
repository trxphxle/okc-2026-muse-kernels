// SPDX-License-Identifier: Apache-2.0
// Diagnostic only: not an attention implementation and NEVER a submission.
// Address contract derived from the approved alygg77 attention producer in
// kernels/muse_decode_layer.py (6cae49a639ee6d14b929a5fd4a63c627f96d7dc0),
// Apache-2.0, built on vjeux/okc-2026 (Apache-2.0). No attention code is replaced.
// Past: row = page_table[b, pos / P] * P + pos % P, pos < seq_lens[b].
// Here P=1, Hkv=2, D=64: K and V each occupy 128 bytes per physical row.
// New rows are NOT baseline cache reads: they come from on-chip kn/vn there.
// The separately labeled include_new mode additionally reads cache_loc[b*L+j].
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint4 read16(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// Eight adjacent lanes load one 128-byte row from each cache. UNROLL separates
// load issue from use: 2*UNROLL independent 16-byte loads per participating lane.
// Warp-local sinks retain two additive moments of EVERY uint32 word, separately
// for K/V and each vector component, plus an exact count of 32-byte K/V pairs.
// No XOR: duplicate words cannot disappear by pair cancellation. Moment 1 is
// sum(word), moment 2 sum(word * (pos*32 + word_in_row + 1)), modulo 2^32.
// Native metadata reports register/local-memory usage and occupancy; the host
// labels this measured checksum-bearing gather rate, not an achieved DRAM rate.
template<int THREADS, int UNROLL, bool NEW>
__global__ __launch_bounds__(THREADS) void round2_gather_read(
    const uint8_t* __restrict__ k, const uint8_t* __restrict__ v,
    const int32_t* __restrict__ pt, const int32_t* __restrict__ seq,
    const int64_t* __restrict__ loc, uint32_t* __restrict__ sink,
    int width, int qlen) {
  constexpr int RPB = THREADS / 8;
  __shared__ int past;
  if (threadIdx.x == 0) past = seq[blockIdx.y];
  __syncthreads();
  const int n = past + (NEW ? qlen : 0);
  const int chunk = threadIdx.x & 7;
  const int leader = (threadIdx.x & 31) & ~7;
  const int first = blockIdx.x * (RPB * UNROLL);
  const int stride = gridDim.x * (RPB * UNROLL);
  uint32_t s[8] = {}, m[8] = {}, count = 0;
  for (int base = first; base < n; base += stride) {
    uint4 keys[UNROLL], vals[UNROLL];
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
      const int pos = base + threadIdx.x / 8 + j * RPB;
      int row = 0;
      if (chunk == 0 && pos < n) {
        if (NEW && pos >= past) row = static_cast<int>(loc[blockIdx.y * qlen + pos - past]);
        else row = pt[static_cast<int64_t>(blockIdx.y) * width + pos];
      }
      // The whole warp participates even when a tail row is inactive.
      row = __shfl_sync(0xffffffffu, row, leader);
      if (pos < n) {
        const size_t off = static_cast<size_t>(row) * 128 + chunk * 16;
        keys[j] = read16(k + off);
        vals[j] = read16(v + off);
      }
    }
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
      const int pos = base + threadIdx.x / 8 + j * RPB;
      if (pos < n) {
        const uint32_t w = static_cast<uint32_t>(pos) * 32 + chunk * 4 + 1;
        const uint32_t a[8] = {keys[j].x, keys[j].y, keys[j].z, keys[j].w,
                               vals[j].x, vals[j].y, vals[j].z, vals[j].w};
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          s[c] += a[c];
          m[c] += a[c] * (w + (c & 3));
        }
        ++count;
      }
    }
  }
  const size_t slot = ((static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x)
                       * (THREADS / 32) + threadIdx.x / 32) * 17;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const uint32_t a = __reduce_add_sync(0xffffffffu, s[c]);
    const uint32_t b = __reduce_add_sync(0xffffffffu, m[c]);
    if ((threadIdx.x & 31) == 0) { sink[slot + c] = a; sink[slot + 8 + c] = b; }
  }
  const uint32_t total = __reduce_add_sync(0xffffffffu, count);
  if ((threadIdx.x & 31) == 0) sink[slot + 16] = total;
}

template<int T, int U, bool N>
void launch(torch::Tensor k, torch::Tensor v, torch::Tensor pt, torch::Tensor seq,
            torch::Tensor loc, torch::Tensor sink, int blocks, int qlen) {
  round2_gather_read<T, U, N><<<dim3(blocks, seq.numel()), T, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(k.data_ptr()), reinterpret_cast<const uint8_t*>(v.data_ptr()),
      pt.data_ptr<int32_t>(), seq.data_ptr<int32_t>(), loc.data_ptr<int64_t>(),
      reinterpret_cast<uint32_t*>(sink.data_ptr<int32_t>()), pt.size(1), qlen);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor k, torch::Tensor v, torch::Tensor pt, torch::Tensor seq,
         torch::Tensor loc, torch::Tensor sink, int64_t geometry, int64_t blocks,
         int64_t qlen, bool include_new) {
  const c10::cuda::CUDAGuard guard(k.device());
  TORCH_CHECK(geometry >= 0 && geometry < 3 && blocks > 0 && blocks < 65536, "bad launch geometry");
  TORCH_CHECK(qlen == 1 || qlen == 8, "diagnostic only supports scored decode/verify");
  TORCH_CHECK(k.is_cuda() && v.device() == k.device() && pt.device() == k.device()
              && seq.device() == k.device() && loc.device() == k.device() && sink.device() == k.device(), "device mismatch");
  TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && pt.is_contiguous() && seq.is_contiguous()
              && loc.is_contiguous() && sink.is_contiguous(), "contiguous storage required");
  TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn && v.scalar_type() == k.scalar_type()
              && k.sizes() == v.sizes() && k.dim() == 4 && k.size(1) == 1
              && k.size(2) == 2 && k.size(3) == 64 && k.size(0) < INT32_MAX, "expected page-1 TP8 FP8 caches");
  TORCH_CHECK(pt.scalar_type() == at::kInt && seq.scalar_type() == at::kInt
              && loc.scalar_type() == at::kLong && sink.scalar_type() == at::kInt, "metadata/sink dtype mismatch");
  const int threads = geometry == 1 ? 128 : 256;
  TORCH_CHECK(seq.dim() == 1 && seq.numel() > 0 && seq.numel() < 65536
              && pt.dim() == 2 && pt.size(0) == seq.numel() && pt.size(1) < INT32_MAX
              && loc.numel() == seq.numel() * qlen
              && sink.numel() == seq.numel() * blocks * (threads / 32) * 17, "metadata/sink shape mismatch");
#define DISPATCH(T,U) if (include_new) launch<T,U,true>(k,v,pt,seq,loc,sink,blocks,qlen); \
                      else launch<T,U,false>(k,v,pt,seq,loc,sink,blocks,qlen)
  if (geometry == 0) { DISPATCH(256,4); }
  else if (geometry == 1) { DISPATCH(128,4); }
  else { DISPATCH(256,2); }
#undef DISPATCH
}

template<int T, int U, bool N> pybind11::dict attributes() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, round2_gather_read<T,U,N>));
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, round2_gather_read<T,U,N>, T, 0));
  pybind11::dict d;
  d["threads"] = T; d["unroll_rows"] = U; d["registers_per_thread"] = a.numRegs;
  d["local_bytes_per_thread"] = a.localSizeBytes; d["static_shared_bytes"] = a.sharedSizeBytes;
  d["max_threads_per_block"] = a.maxThreadsPerBlock; d["active_blocks_per_sm"] = active;
  d["resident_threads_per_sm"] = active * T; d["binary_version"] = a.binaryVersion;
  d["ptx_version"] = a.ptxVersion;
  return d;
}
pybind11::dict info(int64_t geometry, bool include_new) {
#define INFO(T,U) return include_new ? attributes<T,U,true>() : attributes<T,U,false>()
  if (geometry == 0) { INFO(256,4); }
  if (geometry == 1) { INFO(128,4); }
  TORCH_CHECK(geometry == 2, "unknown geometry");
  INFO(256,2);
#undef INFO
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run);
  m.def("info", &info);
}
