// Diagnostic only. WGS TMA addressing/policy and three-slot pipeline derived from
// alygg77/okc-muse-spark-kernels and vjeux/okc-2026 (Apache-2.0).
// Full-byte checksum validation is separate from timed pure-transfer completion.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
constexpr int BYTES = 36864, THREADS = 128, GRID = 132;
__device__ __forceinline__ unsigned smem(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void issue(unsigned char* dst, const unsigned char* src, uint64_t* bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem(bar)), "r"(BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
    :: "r"(smem(dst)), "l"(src), "r"(BYTES), "r"(smem(bar)), "l"(0x12F0000000000000ull) : "memory");
}
__device__ __forceinline__ void wait_slot(uint64_t* bar, unsigned phase) {
  asm volatile("{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra WAIT; }"
    :: "r"(smem(bar)), "r"(phase) : "memory");
}
template <bool CHECK>
__global__ void probe(const unsigned char* data, const int64_t* offsets,
                      const int* bounds, int64_t* result) {
  extern __shared__ __align__(128) unsigned char ring[];
  __shared__ uint64_t bars[3];
  int t = threadIdx.x, begin = bounds[blockIdx.x], end = bounds[blockIdx.x + 1];
  if (t == 0) {
    for (int k = 0; k < 3; ++k)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem(bars+k)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if constexpr (!CHECK) { if (t != 0) return; }
  if (t == 0) {
    for (int k = 0; k < 3 && begin+k < end; ++k) issue(ring+k*BYTES, data+offsets[begin+k], bars+k);
  }
  unsigned long long sum = 0, weighted = 0;
  for (int i = begin; i < end; ++i) {
    int n = i-begin, slot = n%3;
    wait_slot(bars+slot, (n/3)&1);
    if constexpr (CHECK) {
      const unsigned char* p = ring+slot*BYTES;
      for (int k = t; k < BYTES; k += THREADS) {
        unsigned v = p[k]; sum += v; weighted += (unsigned long long)v*(k+1);
      }
      __syncthreads();
    } else {
      // Completion is observed for every full volatile TMA transaction before
      // its slot can be overwritten; no checksum-consumer issue bottleneck.
      sum += BYTES;
      weighted += static_cast<unsigned long long>(offsets[i]);
    }
    if (t == 0 && i+3 < end) {
      if constexpr (CHECK) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue(ring+slot*BYTES, data+offsets[i+3], bars+slot);
    }
  }
  result[(blockIdx.x*THREADS+t)*2] = sum;
  result[(blockIdx.x*THREADS+t)*2+1] = weighted;
}
void run(torch::Tensor data, torch::Tensor offsets, torch::Tensor bounds, torch::Tensor result, bool check) {
  TORCH_CHECK(data.is_cuda() && data.scalar_type()==torch::kUInt8 && data.is_contiguous());
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type()==torch::kInt64 && offsets.is_contiguous());
  TORCH_CHECK(bounds.is_cuda() && bounds.scalar_type()==torch::kInt32 && bounds.numel()==GRID+1);
  TORCH_CHECK(result.is_cuda() && result.scalar_type()==torch::kInt64 && result.numel()==GRID*THREADS*2);
  if (check) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(probe<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 3*BYTES));
    probe<true><<<GRID, THREADS, 3*BYTES, at::cuda::getCurrentCUDAStream()>>>(data.data_ptr<unsigned char>(), offsets.data_ptr<int64_t>(), bounds.data_ptr<int>(), result.data_ptr<int64_t>());
  } else {
    C10_CUDA_CHECK(cudaFuncSetAttribute(probe<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 3*BYTES));
    probe<false><<<GRID, THREADS, 3*BYTES, at::cuda::getCurrentCUDAStream()>>>(data.data_ptr<unsigned char>(), offsets.data_ptr<int64_t>(), bounds.data_ptr<int>(), result.data_ptr<int64_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
