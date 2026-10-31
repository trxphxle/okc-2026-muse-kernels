// Original experts-v32 diagnostic, not a submission.
// Three-slot bulk-TMA ring follows the approved vjeux/alygg77 MoE pattern
// (Apache-2.0); pairing, multicast protocol, and validation are lane work.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace reuse_probe {
namespace cg = cooperative_groups;
constexpr int THREADS = 256, GRID = 132, PAIRS = GRID / 2;
constexpr int SB = 36864, WORDS = SB / 4, SLOTS = 3, UNIT_STAGES = 16;
constexpr int SMEM = 232192;  // Reserve native-WGS-sized storage: one CTA/SM.
constexpr unsigned A = 1664525u, B = 1013904223u;

__device__ __forceinline__ uint32_t shared_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void expect(uint64_t* b) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(shared_addr(b)), "r"(SB) : "memory");
}
__device__ __forceinline__ void wait_full(uint64_t* b, unsigned parity) {
  asm volatile("{ .reg .pred p; wait_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra wait_%=; }\n"
               :: "r"(shared_addr(b)), "r"(parity) : "memory");
}

template<int MODE>
__device__ __forceinline__ void issue(const unsigned* source, int2 item, int rank,
                                     int stage, unsigned char* dst, uint64_t* barrier,
                                     unsigned& issued) {
  const int unit = rank ? item.y : item.x;
  if (unit < 0) return;
  const unsigned* src = source + (static_cast<size_t>(unit) * UNIT_STAGES + stage) * WORDS;
  if constexpr (MODE == 2) {
    if (item.x == item.y) {
      if (rank == 0) {
        const uint16_t mask = 3;
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;\n"
                     :: "r"(shared_addr(dst)), "l"(src), "r"(SB), "r"(shared_addr(barrier)), "h"(mask) : "memory");
        ++issued;
      }
      return;
    }
  }
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(shared_addr(dst)), "l"(src), "r"(SB), "r"(shared_addr(barrier)) : "memory");
  ++issued;
}

// MODE0: unclustered independent. MODE1: matched cluster protocol, independent
// copies. MODE2: multicast identical pair sources; singleton tasks stay independent.
// CHECK counts every mismatching word; timed builds observe stage endpoints and
// completion counts. All TMA requests remain volatile in both builds.
template<int MODE, bool CHECK>
__global__ void __launch_bounds__(THREADS, 1) stream(const unsigned* source, const int2* plan,
                                                    int groups, unsigned long long* output) {
  extern __shared__ __align__(128) unsigned char storage[];
  uint64_t* full = reinterpret_cast<uint64_t*>(storage);
  unsigned* warp_sums = reinterpret_cast<unsigned*>(storage + 32);
  unsigned char* ring = storage + 128;
  const int tid = threadIdx.x, rank = blockIdx.x & 1, pair = blockIdx.x >> 1;
  const int tasks = pair < groups ? (groups - 1 - pair) / PAIRS + 1 : 0;
  const int steps = tasks * UNIT_STAGES;
  auto cluster = cg::this_cluster();
  unsigned issued = 0, consumed = 0, sum = 0;
  if (tid == 0) {
    for (int slot = 0; slot < SLOTS; ++slot)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(shared_addr(full + slot)) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    for (int step = 0; step < SLOTS && step < steps; ++step) {
      const int2 item = plan[pair];
      if ((rank ? item.y : item.x) >= 0) expect(full + step);
    }
  }
  __syncthreads();
  if constexpr (MODE != 0) cluster.sync();
  if (tid == 0) {
    for (int step = 0; step < SLOTS && step < steps; ++step)
      issue<MODE>(source, plan[pair], rank, step, ring + step * SB, full + step, issued);
  }
  for (int step = 0; step < steps; ++step) {
    const int slot = step % SLOTS;
    const int2 item = plan[pair + (step / UNIT_STAGES) * PAIRS];
    const int unit = rank ? item.y : item.x;
    if (tid == 0 && unit >= 0) wait_full(full + slot, (step / SLOTS) & 1);
    __syncthreads();
    if (unit >= 0) {
      const unsigned* data = reinterpret_cast<const unsigned*>(ring + slot * SB);
      if constexpr (CHECK) {
        const size_t first = (static_cast<size_t>(unit) * UNIT_STAGES + step % UNIT_STAGES) * WORDS;
        for (int i = tid; i < WORDS; i += THREADS)
          sum += data[i] != (static_cast<unsigned>(first + i) * A + B);
      } else if (tid == 0) {
        sum += data[0] + data[WORDS - 1];
      }
      ++consumed;
    }
    __syncthreads();
    const int next = step + SLOTS;
    if (tid == 0 && next < steps) {
      const int2 next_item = plan[pair + (next / UNIT_STAGES) * PAIRS];
      if ((rank ? next_item.y : next_item.x) >= 0) expect(full + slot);
    }
    // Both CTAs have consumed this slot and installed their next expected-byte
    // count before the leader can multicast into either CTA's storage again.
    if constexpr (MODE != 0) cluster.sync();
    if (tid == 0 && next < steps)
      issue<MODE>(source, plan[pair + (next / UNIT_STAGES) * PAIRS], rank, next % UNIT_STAGES,
                  ring + slot * SB, full + slot, issued);
  }
  if constexpr (MODE != 0) cluster.sync();
  sum = __reduce_add_sync(0xffffffffu, sum);
  if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
  __syncthreads();
  if (tid < 32) {
    sum = tid < THREADS / 32 ? warp_sums[tid] : 0;
    sum = __reduce_add_sync(0xffffffffu, sum);
    if (tid == 0) {
      output[blockIdx.x * 3] = sum;
      output[blockIdx.x * 3 + 1] = consumed;
      output[blockIdx.x * 3 + 2] = issued;
    }
  }
}

__global__ void fill(unsigned* data, size_t words) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    data[i] = static_cast<unsigned>(i) * A + B;
}

template<int MODE, bool CHECK>
void launch(torch::Tensor source, torch::Tensor plan, torch::Tensor output) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(GRID);
  cfg.blockDim = dim3(THREADS);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  if constexpr (MODE != 0) {
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
    cfg.attrs = &attr; cfg.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, stream<MODE, CHECK>,
      reinterpret_cast<const unsigned*>(source.data_ptr()),
      reinterpret_cast<const int2*>(plan.data_ptr()), static_cast<int>(plan.size(0)),
      reinterpret_cast<unsigned long long*>(output.data_ptr())));
}
}  // namespace reuse_probe

std::vector<int64_t> configure() {
  using namespace reuse_probe;
  int device; C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop; C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  TORCH_CHECK(prop.major == 9 && prop.multiProcessorCount == GRID, "Probe requires the 132-SM H200 lane GPU");
#define CONFIGURE(M, C) C10_CUDA_CHECK(cudaFuncSetAttribute(stream<M, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM))
  CONFIGURE(0, false); CONFIGURE(1, false); CONFIGURE(2, false);
  CONFIGURE(0, true); CONFIGURE(1, true); CONFIGURE(2, true);
#undef CONFIGURE
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(GRID); cfg.blockDim = dim3(THREADS); cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute attr{}; attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = 2; attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
  cfg.attrs = &attr; cfg.numAttrs = 1;
  int active = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&active, stream<2, false>, &cfg));
  return {prop.multiProcessorCount, SMEM, active};
}

void initialize(torch::Tensor source) {
  reuse_probe::fill<<<4096, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned*>(source.data_ptr()), source.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor source, torch::Tensor plan, torch::Tensor output, int64_t mode, bool check) {
  using namespace reuse_probe;
  TORCH_CHECK(plan.dim() == 2 && plan.size(1) == 2 && plan.is_contiguous(), "Bad read plan");
  TORCH_CHECK(output.numel() == GRID * 3, "Bad output size");
#define DISPATCH(M) if (check) launch<M, true>(source, plan, output); else launch<M, false>(source, plan, output)
  if (mode == 0) { DISPATCH(0); }
  else if (mode == 1) { DISPATCH(1); }
  else { TORCH_CHECK(mode == 2, "Bad mode"); DISPATCH(2); }
#undef DISPATCH
}
