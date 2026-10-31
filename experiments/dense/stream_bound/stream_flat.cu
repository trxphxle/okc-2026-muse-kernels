// SPDX-License-Identifier: Apache-2.0
// Diagnostic only. Same physical chunks as stream_kernel.cu, one chunk/stage.
// TMA/mbarrier patterns: Copyright vjeux (f35a41e), Apache-2.0, and
// Copyright2026 alygg77 (6cae49a), Apache-2.0. No layer computation.
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace dense_stream_flat {
constexpr int Threads=32,Slots=4,StageBytes=32768,Dynamic=Slots*StageBytes;
__device__ __forceinline__ uint32_t addr(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint64_t timer() { uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");return t; }
__device__ __forceinline__ void wait(uint32_t b,uint32_t phase) {
  asm volatile("{ .reg .pred p; again_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra again_%=; }"::"r"(b),"r"(phase):"memory");
}
__device__ __forceinline__ void issue(const int64_t* chunks,int64_t index,int slot,unsigned char* data,uint64_t* bars,int* sizes) {
  if(threadIdx.x==0) {
    const int64_t* item=chunks+index*3;
    const uint64_t source=uint64_t(__ldg(item));
    const uint32_t bytes=uint32_t(__ldg(item+1));
    sizes[slot]=bytes;
    const uint32_t b=addr(bars+slot),d=addr(data+slot*StageBytes);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(b),"r"(bytes):"memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(d),"l"(source),"r"(bytes),"r"(b):"memory");
  }
  __syncwarp();
}
template<bool Validate>
__global__ __launch_bounds__(Threads,1) void stream_kernel(const int64_t* chunks,int64_t* sink,int64_t* stamps,int count) {
  extern __shared__ __align__(128) unsigned char data[];
  __shared__ __align__(8) uint64_t bars[Slots];
  __shared__ int sizes[Slots];
  const int lane=threadIdx.x,b=blockIdx.x;
  const int64_t n=count>b?1+(count-1-b)/gridDim.x:0;
  if(lane==0) {
    if(stamps) { stamps[4*b]=clock64();stamps[4*b+2]=timer(); }
    #pragma unroll
    for(int s=0;s<Slots;++s) asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(addr(bars+s)):"memory");
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncwarp();
  #pragma unroll
  for(int s=0;s<Slots;++s) if(s<n) issue(chunks,b+int64_t(s)*gridDim.x,s,data,bars,sizes);
  uint64_t bytes=0;uint32_t x=0;
  for(int64_t i=0;i<n;++i) {
    const int s=i%Slots;
    wait(addr(bars+s),uint32_t((i/Slots)&1));
    const int size=sizes[s];
    if constexpr(Validate) {
      const volatile uint32_t* p=reinterpret_cast<const volatile uint32_t*>(data+s*StageBytes);
      for(int j=lane;j<size/4;j+=Threads) x^=p[j];
    }
    if(lane==0) bytes+=size;
    __syncwarp();
    if(i+Slots<n) issue(chunks,b+(i+Slots)*gridDim.x,s,data,bars,sizes);
  }
  if constexpr(Validate) {
    #pragma unroll
    for(int d=16;d;d/=2) x^=__shfl_xor_sync(0xffffffff,x,d);
  }
  if(lane==0) {
    sink[4*b]=bytes;sink[4*b+1]=x;sink[4*b+2]=n;sink[4*b+3]=n;
    if(stamps) { stamps[4*b+1]=clock64();stamps[4*b+3]=timer(); }
  }
}
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<bool V> std::vector<int64_t> info() {
  int device;ck(cudaGetDevice(&device));cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,device));
  ck(cudaFuncSetAttribute(stream_kernel<V>,cudaFuncAttributeMaxDynamicSharedMemorySize,Dynamic));
  cudaFuncAttributes a{};ck(cudaFuncGetAttributes(&a,stream_kernel<V>));
  int active;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,stream_kernel<V>,Threads,Dynamic));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,Dynamic,Threads,active,p.multiProcessorCount,StageBytes,Slots};
}
}
std::vector<int64_t> stream_info(int mode) {
  if(mode==0)return dense_stream_flat::info<false>();
  if(mode==1)return dense_stream_flat::info<true>();
  throw std::invalid_argument("mode must be0 or1");
}
void stream_launch(int64_t chunks,int64_t stages,int64_t sink,int64_t stamps,int nstages,int blocks,int mode,int64_t stream) {
  using namespace dense_stream_flat;
  if(nstages<0||blocks<=0||!sink||(sink&7)||(stamps&7)||(nstages&&(!chunks||(chunks&7)))||mode<0||mode>1)throw std::invalid_argument("invalid flat stream arguments");
  const auto st=reinterpret_cast<cudaStream_t>(stream);
  if(mode) stream_kernel<true><<<blocks,Threads,Dynamic,st>>>(reinterpret_cast<const int64_t*>(chunks),reinterpret_cast<int64_t*>(sink),reinterpret_cast<int64_t*>(stamps),nstages);
  else stream_kernel<false><<<blocks,Threads,Dynamic,st>>>(reinterpret_cast<const int64_t*>(chunks),reinterpret_cast<int64_t*>(sink),reinterpret_cast<int64_t*>(stamps),nstages);
  ck(cudaGetLastError());
}
