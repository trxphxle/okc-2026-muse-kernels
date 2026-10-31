// Diagnostic VMM backing only; no expert/dense arithmetic or weight format edits.
// CUDA Driver API VMM and CUmemAllocationProp contracts are documented by NVIDIA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <memory>

namespace py = pybind11;
static std::atomic<int64_t> live_count{0}, live_bytes{0}, cleanup_errors{0};

static void check_driver(CUresult status, const char* operation) {
  if (status == CUDA_SUCCESS) return;
  const char *name = nullptr, *description = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &description);
  TORCH_CHECK(false, operation, ": ", name ? name : "unknown CUDA status", " (", int(status), ") ",
              description ? description : "");
}

static CUmemAllocationProp properties(int device, bool compression) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = compression ? CU_MEM_ALLOCATION_COMP_GENERIC : CU_MEM_ALLOCATION_COMP_NONE;
  return prop;
}

static size_t granularity(const CUmemAllocationProp& prop, CUmemAllocationGranularity_flags mode) {
  size_t result = 0;
  check_driver(cuMemGetAllocationGranularity(&result, &prop, mode), "cuMemGetAllocationGranularity");
  TORCH_CHECK(result && !(result & (result - 1)), "Expected power-of-two VMM granularity");
  return result;
}

static py::dict capabilities() {
  int device = 0, driver_version = 0, runtime_version = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  check_driver(cuInit(0), "cuInit");
  check_driver(cuDriverGetVersion(&driver_version), "cuDriverGetVersion");
  C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUdevice cu_device;
  check_driver(cuDeviceGet(&cu_device, device), "cuDeviceGet");
  int compression = 0, vmm = 0;
  check_driver(cuDeviceGetAttribute(&compression, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, cu_device),
               "cuDeviceGetAttribute(GENERIC_COMPRESSION_SUPPORTED)");
  check_driver(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cu_device),
               "cuDeviceGetAttribute(VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED)");
  cudaDeviceProp info{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&info, device));
  py::dict result;
  result["device"] = device;
  result["gpu"] = info.name;
  result["sm_count"] = info.multiProcessorCount;
  result["driver_version"] = driver_version;
  result["runtime_version"] = runtime_version;
  result["generic_compression_supported"] = bool(compression);
  result["vmm_supported"] = bool(vmm);
  const auto prop = properties(device, false);
  if (vmm) {
    result["normal_min_granularity"] = granularity(prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    result["normal_recommended_granularity"] = granularity(prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  }
  return result;
}

struct Mapping {
  CUmemGenericAllocationHandle handle = 0;
  CUdeviceptr address = 0;
  size_t bytes = 0;
  bool mapped = false, counted = false;
  ~Mapping() noexcept {
    auto release = [](CUresult status) { if (status != CUDA_SUCCESS) ++cleanup_errors; };
    if (mapped) release(cuMemUnmap(address, bytes));
    if (handle) release(cuMemRelease(handle));
    if (address) release(cuMemAddressFree(address, bytes));
    if (counted) { --live_count; live_bytes -= static_cast<int64_t>(bytes); }
  }
};

static py::tuple allocate(int64_t logical_bytes, bool compression, bool recommended) {
  TORCH_CHECK(logical_bytes > 0, "A nonempty allocation is required");
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const auto prop = properties(device, compression);
  size_t minimum = granularity(prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  size_t preferred = granularity(prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  size_t alignment = recommended ? preferred : minimum;
  const size_t n = static_cast<size_t>(logical_bytes);
  TORCH_CHECK(n <= SIZE_MAX - alignment + 1, "Allocation size overflow");
  auto owner = std::make_shared<Mapping>();
  owner->bytes = ((n + alignment - 1) / alignment) * alignment;
  check_driver(cuMemCreate(&owner->handle, owner->bytes, &prop, 0), "cuMemCreate");
  CUmemAllocationProp granted{};
  check_driver(cuMemGetAllocationPropertiesFromHandle(&granted, owner->handle),
               "cuMemGetAllocationPropertiesFromHandle");
  check_driver(cuMemAddressReserve(&owner->address, owner->bytes, alignment, 0, 0), "cuMemAddressReserve");
  check_driver(cuMemMap(owner->address, owner->bytes, 0, owner->handle, 0), "cuMemMap");
  owner->mapped = true;
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  check_driver(cuMemSetAccess(owner->address, owner->bytes, &access, 1), "cuMemSetAccess");
  owner->counted = true;
  ++live_count;
  live_bytes += static_cast<int64_t>(owner->bytes);
  auto tensor = torch::from_blob(reinterpret_cast<void*>(owner->address), {logical_bytes},
      [owner](void*) mutable { owner.reset(); }, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));
  py::dict metadata;
  metadata["device"] = device;
  metadata["compression_requested"] = int(prop.allocFlags.compressionType);
  metadata["compression_requested_is_generic"] = prop.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC;
  metadata["compression_granted"] = int(granted.allocFlags.compressionType);
  metadata["compression_granted_is_generic"] = granted.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC;
  metadata["minimum_granularity"] = minimum;
  metadata["recommended_granularity"] = preferred;
  metadata["alignment"] = alignment;
  metadata["address"] = static_cast<uint64_t>(owner->address);
  metadata["address_mod_recommended"] = static_cast<uint64_t>(owner->address % preferred);
  metadata["logical_bytes"] = logical_bytes;
  metadata["allocation_bytes"] = owner->bytes;
  metadata["mapped_bytes"] = owner->bytes;
  metadata["granted_location_type"] = int(granted.location.type);
  metadata["granted_location_id"] = granted.location.id;
  return py::make_tuple(tensor, metadata);
}

__global__ void copy_bits(unsigned char* destination, const unsigned char* source, size_t bytes) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t words = bytes / sizeof(uint4);
  if (i < words) reinterpret_cast<uint4*>(destination)[i] = reinterpret_cast<const uint4*>(source)[i];
  if (i == words) for (size_t p = words * sizeof(uint4); p < bytes; ++p) destination[p] = source[p];
}

static void copy_into(torch::Tensor destination, torch::Tensor source) {
  TORCH_CHECK(destination.is_cuda() && source.is_cuda() && destination.device() == source.device());
  TORCH_CHECK(destination.is_contiguous() && source.is_contiguous());
  size_t bytes = source.numel() * source.element_size();
  TORCH_CHECK(bytes == destination.numel() * destination.element_size() && bytes > 0);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(destination.data_ptr()) % 16) == 0 &&
              (reinterpret_cast<uintptr_t>(source.data_ptr()) % 16) == 0, "16-byte copy alignment required");
  c10::cuda::CUDAGuard guard(source.device());
  copy_bits<<<(bytes / 16 + 1 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<unsigned char*>(destination.data_ptr()), static_cast<const unsigned char*>(source.data_ptr()), bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

constexpr int READ_BYTES = 32768, READ_SLOTS = 4;
__device__ __forceinline__ unsigned shared_address(const void* pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__global__ void read_stream(const unsigned char* source, size_t bytes, int64_t* completed) {
  extern __shared__ __align__(128) unsigned char ring[];
  __shared__ uint64_t barriers[READ_SLOTS];
  if (threadIdx.x != 0) return;
  const size_t stages = (bytes + READ_BYTES - 1) / READ_BYTES;
  const size_t first = stages * blockIdx.x / gridDim.x;
  const size_t last = stages * (blockIdx.x + 1) / gridDim.x;
  for (int slot = 0; slot < READ_SLOTS; ++slot)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(shared_address(barriers + slot)) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  auto stage_bytes = [&](size_t stage) {
    size_t left = bytes - stage * READ_BYTES;
    return static_cast<unsigned>(left < READ_BYTES ? left : READ_BYTES);
  };
  auto issue = [&](size_t stage, int slot) {
    if (stage >= last) return;
    unsigned transfer = stage_bytes(stage) & ~15u;
    if (!transfer) return;
    unsigned bar = shared_address(barriers + slot);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(transfer) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(shared_address(ring + slot * READ_BYTES)), "l"(source + stage * READ_BYTES),
           "r"(transfer), "r"(bar) : "memory");
  };
  for (int slot = 0; slot < READ_SLOTS; ++slot) issue(first + slot, slot);
  size_t total = 0;
  unsigned tail_checksum = 0;
  for (size_t stage = first; stage < last; ++stage) {
    int slot = (stage - first) % READ_SLOTS;
    unsigned amount = stage_bytes(stage), transfer = amount & ~15u;
    if (transfer) {
      unsigned parity = ((stage - first) / READ_SLOTS) & 1;
      asm volatile("{ .reg .pred ready; WAIT: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1; @!ready bra WAIT; }"
          :: "r"(shared_address(barriers + slot)), "r"(parity) : "memory");
    }
    for (unsigned p = transfer; p < amount; ++p)
      tail_checksum ^= reinterpret_cast<const volatile unsigned char*>(source)[stage * READ_BYTES + p];
    total += amount;
    issue(stage + READ_SLOTS, slot);
  }
  // Volatile transfers and completed mbarrier transactions cannot be removed.
  // Tail loads are volatile too; preserve their checksum without changing count.
  asm volatile("" :: "r"(tail_checksum) : "memory");
  completed[blockIdx.x] = static_cast<int64_t>(total);
}

static void read_bytes(torch::Tensor source, torch::Tensor completed) {
  TORCH_CHECK(source.is_cuda() && source.is_contiguous());
  TORCH_CHECK(completed.is_cuda() && completed.device() == source.device() && completed.is_contiguous()
              && completed.scalar_type() == torch::kInt64 && completed.dim() == 1);
  TORCH_CHECK(completed.numel() > 0 && source.numel() > 0);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(source.data_ptr()) % 16) == 0, "Read base must be 16-byte aligned");
  c10::cuda::CUDAGuard guard(source.device());
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(read_stream, cudaFuncAttributeMaxDynamicSharedMemorySize, READ_BYTES * READ_SLOTS));
    configured = true;
  }
  read_stream<<<completed.numel(), 32, READ_BYTES * READ_SLOTS, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const unsigned char*>(source.data_ptr()), source.numel() * source.element_size(), completed.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static py::dict live_stats() {
  py::dict result;
  result["allocations"] = live_count.load();
  result["bytes"] = live_bytes.load();
  result["cleanup_errors"] = cleanup_errors.load();
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("capabilities", &capabilities);
  module.def("allocate", &allocate, py::arg("nbytes"), py::arg("compress"), py::arg("recommended") = true);
  module.def("copy_into", &copy_into);
  module.def("read_bytes", &read_bytes);
  module.def("live_stats", &live_stats);
}
