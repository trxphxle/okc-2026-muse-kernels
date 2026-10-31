// SPDX-License-Identifier: Apache-2.0
// Ceiling diagnostic only: no decode arithmetic or production megakernel.
// TMA/mbarrier helpers follow the approved work/bench/d1_stream_modes.cu
// tma2_kernel and vjeux/alygg77 bulk-copy ring patterns.
// Copyright vjeux, https://github.com/vjeux/okc-2026 (f35a41e), Apache-2.0.
// Copyright 2026 alygg77, https://github.com/alygg77/okc-muse-spark-kernels
// (6cae49a639ee6d14b929a5fd4a63c627f96d7dc0), Apache-2.0.
// No own-MoE prologue, routing, math, or external implementation is included.

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dense_stream_bound {
constexpr int kThreads = 32;
constexpr int kStageBytes = 32768;
constexpr int kSlots = 4;
constexpr int kDynamicBytes = kSlots * kStageBytes;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ __forceinline__ uint32_t shared_address(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t global_timer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

__device__ __forceinline__ void await_stage(uint32_t barrier, uint32_t phase) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n "
      "@!p bra WAIT_%=;\n}\n"
      :: "r"(barrier), "r"(phase) : "memory");
}

// All 32 lanes call this function together. One arrival accounts for the
// complete stage, BEFORE any lane issues its disjoint subset of transactions.
// Descriptors are immutable CUDA int64 triples; the planner checks bounds,
// 16-byte alignment and exact contiguous packing (no undeclared slot holes).
__device__ __forceinline__ void issue_stage(
    const int64_t* chunks, const int64_t* stages, int64_t global_stage,
    int slot, uint8_t* memory, uint64_t* barriers,
    int* slot_bytes, int* slot_chunks) {
  const int lane = threadIdx.x;
  long long first = 0;  // Explicit CUDA 64-bit shuffle overload on LP64 hosts.
  int count = 0;
  int bytes = 0;
  if (lane == 0) {
    const int64_t* descriptor = stages + global_stage * 3;
    first = __ldg(descriptor);
    count = static_cast<int>(__ldg(descriptor + 1));
    bytes = static_cast<int>(__ldg(descriptor + 2));
    slot_bytes[slot] = bytes;
    slot_chunks[slot] = count;
  }
  first = __shfl_sync(kWarpMask, first, 0);
  count = __shfl_sync(kWarpMask, count, 0);
  bytes = __shfl_sync(kWarpMask, bytes, 0);
  const uint32_t barrier = shared_address(barriers + slot);
  if (lane == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(barrier), "r"(bytes) : "memory");
  }
  __syncwarp(kWarpMask);
  const uint32_t destination = shared_address(memory + slot * kStageBytes);
  // A 256-page KV stage takes eight warp issue iterations, not 256 serial
  // elected-lane issues. Dense 32-KiB stages still use just one transaction.
  for (int chunk = lane; chunk < count; chunk += kThreads) {
    const int64_t* descriptor = chunks + (first + chunk) * 3;
    const uint64_t source = static_cast<uint64_t>(__ldg(descriptor));
    const uint32_t size = static_cast<uint32_t>(__ldg(descriptor + 1));
    const uint32_t offset = static_cast<uint32_t>(__ldg(descriptor + 2));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :: "r"(destination + offset), "l"(source), "r"(size), "r"(barrier)
        : "memory");
  }
  __syncwarp(kWarpMask);
}

// Exactly one persistent grid covers the entire mixed-category stream.
// CTA b owns stages b, b+gridDim.x, ...; every slot has a separate parity.
// Mode 0 deliberately does not read the shared payload. Volatile transfers,
// completion waits and completed-byte sinks are retained; NCU DRAM-byte proof
// is REQUIRED before interpreting its timing. Mode 1 drains every uint32 word
// for exact XOR validation and is NOT the reported bandwidth lower bound.
template<bool Validate>
__global__ __launch_bounds__(kThreads, 1) void stream_kernel(
    const int64_t* chunks, const int64_t* stages, int64_t* sink,
    int64_t* stamps, int nstages) {
  extern __shared__ __align__(128) uint8_t memory[];
  __shared__ __align__(8) uint64_t barriers[kSlots];
  __shared__ int slot_bytes[kSlots];
  __shared__ int slot_chunks[kSlots];
  const int lane = threadIdx.x;
  const int block = blockIdx.x;
  const int stride = gridDim.x;
  const int64_t count = nstages > block ? 1 + (nstages - 1 - block) / stride : 0;
  if (lane == 0) {
    if (stamps != nullptr) {
      stamps[static_cast<int64_t>(block) * 4] = clock64();
      stamps[static_cast<int64_t>(block) * 4 + 2] = global_timer();
    }
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
      const uint32_t barrier = shared_address(barriers + slot);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                   :: "r"(barrier) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncwarp(kWarpMask);
#pragma unroll
  for (int slot = 0; slot < kSlots; ++slot) {
    if (slot < count) {
      issue_stage(chunks, stages, block + static_cast<int64_t>(slot) * stride,
                  slot, memory, barriers, slot_bytes, slot_chunks);
    }
  }

  uint64_t completed_bytes = 0;
  uint64_t completed_chunks = 0;
  uint32_t checksum = 0;
  for (int64_t local_stage = 0; local_stage < count; ++local_stage) {
    const int slot = static_cast<int>(local_stage % kSlots);
    const uint32_t phase = static_cast<uint32_t>((local_stage / kSlots) & 1);
    // Every lane observes completion, including the validation readers.
    await_stage(shared_address(barriers + slot), phase);
    const int bytes = slot_bytes[slot];
    if constexpr (Validate) {
      const volatile uint32_t* payload =
          reinterpret_cast<const volatile uint32_t*>(memory + slot * kStageBytes);
      for (int word = lane; word < bytes / 4; word += kThreads) {
        checksum ^= payload[word];
      }
    }
    if (lane == 0) {
      completed_bytes += bytes;
      completed_chunks += slot_chunks[slot];
    }
    // No producer can reuse this slot before all waits/readers have finished.
    __syncwarp(kWarpMask);
    const int64_t next = local_stage + kSlots;
    if (next < count) {
      issue_stage(chunks, stages, block + next * stride, slot,
                  memory, barriers, slot_bytes, slot_chunks);
    }
  }
  if constexpr (Validate) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      checksum ^= __shfl_xor_sync(kWarpMask, checksum, offset);
    }
  }
  if (lane == 0) {
    const int64_t output = static_cast<int64_t>(block) * 4;
    // int64[blocks,4]: physical bytes, zero-extended uint32 XOR, stages, chunks.
    // All four entries are zero for an empty CTA, including nstages == 0.
    sink[output] = static_cast<int64_t>(completed_bytes);
    sink[output + 1] = static_cast<int64_t>(checksum);
    sink[output + 2] = count;
    sink[output + 3] = static_cast<int64_t>(completed_chunks);
    if (stamps != nullptr) {
      stamps[output + 1] = clock64();
      stamps[output + 3] = global_timer();
    }
  }
}

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void check_mode(int mode) {
  if (mode != 0 && mode != 1) throw std::invalid_argument("stream mode must be 0 or 1");
}

template<bool Validate>
std::vector<int64_t> resource_info() {
  int device;
  check_cuda(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 9) throw std::runtime_error("stream requires SM90 or newer");
  // Untimed setup: build() initializes both specializations before capture.
  check_cuda(cudaFuncSetAttribute(stream_kernel<Validate>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicBytes));
  cudaFuncAttributes attributes{};
  check_cuda(cudaFuncGetAttributes(&attributes, stream_kernel<Validate>));
  int active_blocks;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, stream_kernel<Validate>, kThreads, kDynamicBytes));
  return {attributes.numRegs, static_cast<int64_t>(attributes.sharedSizeBytes),
          static_cast<int64_t>(attributes.localSizeBytes), attributes.maxThreadsPerBlock,
          kDynamicBytes, kThreads, active_blocks, properties.multiProcessorCount,
          kStageBytes, kSlots};
}
}  // namespace dense_stream_bound

std::vector<int64_t> stream_info(int mode) {
  dense_stream_bound::check_mode(mode);
  return mode == 0 ? dense_stream_bound::resource_info<false>()
                   : dense_stream_bound::resource_info<true>();
}

void stream_launch(int64_t chunks, int64_t stages, int64_t sink, int64_t stamps,
                   int nstages, int blocks, int mode, int64_t cuda_stream) {
  using namespace dense_stream_bound;
  check_mode(mode);
  if (nstages < 0 || blocks <= 0 || sink == 0 || (sink & 7) || (stamps & 7) ||
      (nstages != 0 && (chunks == 0 || stages == 0 || (chunks & 7) || (stages & 7)))) {
    throw std::invalid_argument("invalid stream dimensions or descriptor/output pointers");
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (mode == 0) {
    stream_kernel<false><<<blocks, kThreads, kDynamicBytes, stream>>>(
        reinterpret_cast<const int64_t*>(chunks), reinterpret_cast<const int64_t*>(stages),
        reinterpret_cast<int64_t*>(sink), reinterpret_cast<int64_t*>(stamps), nstages);
  } else {
    stream_kernel<true><<<blocks, kThreads, kDynamicBytes, stream>>>(
        reinterpret_cast<const int64_t*>(chunks), reinterpret_cast<const int64_t*>(stages),
        reinterpret_cast<int64_t*>(sink), reinterpret_cast<int64_t*>(stamps), nstages);
  }
  // Asynchronous launch-error check only: no stream/device synchronization,
  // allocations, descriptor construction or input preprocessing in this call.
  check_cuda(cudaGetLastError());
}
