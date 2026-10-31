// Diagnostic only: no GEMV, no globally expanded weights, no escape correction.
// decode8/prmt/lop3_sel adapted verbatim from the approved vjeux/okc-2026
// Apache-2.0 implementation as maintained by alygg77/okc-muse-spark-kernels,
// pinned source work/prior-art/alygg77/kernels/muse_bf16_proj.py:196-218.
// Portions Copyright vjeux, https://github.com/vjeux/okc-2026 (commit f35a41e),
// licensed under Apache License2.0. Submitted by alygg77 under Apache
// License2.0 with this notice retained.
// License: https://www.apache.org/licenses/LICENSE-2.0
// The surrounding diagnostic and window4 decoder are new lane code.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
#include <vector>

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}
__device__ __forceinline__ void decode11(uint32_t a, uint32_t b, uint32_t c, uint32_t base, uint32_t* r) {
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t lo = j < 2 ? a : b;
    const uint32_t c0 = (c >> (6*j)) & 7, c1 = (c >> (6*j+3)) & 7;
    const uint32_t h0 = (base + (c0 & 3)) | ((c0 & 4) << 5);
    const uint32_t h1 = (base + (c1 & 3)) | ((c1 & 4) << 5);
    r[j] = ((lo >> (16*(j&1))) & 255) | (h0 << 8)
         | (((lo >> (16*(j&1)+8)) & 255) << 16) | (h1 << 24);
  }
}
__device__ __forceinline__ unsigned long long observed(uint32_t* r) {
  unsigned long long s = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j) s += (r[j] & 65535u) + (r[j] >> 16);
  return s;
}

// Mode 0 reads every uint32, including any serialized escape side tables.
// Modes 11/12 decode the main payload only, returning a sum of BF16 bit codes.
// cycles[] is per-CTA elapsed clock64; no clock-frequency inference is needed.
template<int Mode, bool Registers>
__global__ void probe(const uint8_t* data, uint64_t n, int base, int repeats,
                      unsigned long long* sums, unsigned long long* cycles) {
  const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = gridDim.x * blockDim.x;
  unsigned long long sum = 0;
  uint32_t a=0, b=0, c=0;
  if constexpr (Registers) {
    const uint64_t g = tid % (n/8);
    const uint2 low = reinterpret_cast<const uint2*>(data)[g];
    a=low.x; b=low.y;
    if constexpr (Mode == 12 || Mode == 0) c = reinterpret_cast<const uint32_t*>(data+n)[g];
    else c = uint32_t(data[n+3*g]) | (uint32_t(data[n+3*g+1])<<8) | (uint32_t(data[n+3*g+2])<<16);
  }
  __syncthreads();
  const unsigned long long start = clock64();
  if constexpr (Registers) {
    #pragma unroll 1
    for (int i=0; i<repeats; ++i) {
      // Opaque identity makes each iteration's input observable and prevents
      // loop-invariant decode hoisting, without an instruction or memory load.
      asm volatile("" : "+r"(a), "+r"(b), "+r"(c));
      if constexpr (Mode == 0) sum += uint64_t(a) + b + c;
      else {
        uint32_t r[4];
        if constexpr (Mode == 12) decode8(a,b,c,uint32_t(base)*0x01010101u,r);
        else decode11(a,b,c,base,r);
        sum += observed(r);
      }
    }
  } else if constexpr (Mode == 0) {
    for (uint64_t j=tid; j<(n+3)/4; j+=stride) {
      const uint64_t offset=4*j;
      uint32_t v=0;
      if (offset+4<=n) v=reinterpret_cast<const uint32_t*>(data)[j];
      else for (int k=0;k<4 && offset+k<n;++k) v |= uint32_t(data[offset+k])<<(8*k);
      sum += v;
    }
  } else {
    for (uint64_t g=tid; g<n/8; g+=stride) {
      const uint2 low=reinterpret_cast<const uint2*>(data)[g];
      uint32_t code;
      if constexpr (Mode == 12) code=reinterpret_cast<const uint32_t*>(data+n)[g];
      else code=uint32_t(data[n+3*g]) | (uint32_t(data[n+3*g+1])<<8) | (uint32_t(data[n+3*g+2])<<16);
      uint32_t r[4];
      if constexpr (Mode == 12) decode8(low.x,low.y,code,uint32_t(base)*0x01010101u,r);
      else decode11(low.x,low.y,code,base,r);
      sum += observed(r);
    }
  }
  __syncthreads();
  const unsigned long long end=clock64();
  sums[tid]=sum;
  if (threadIdx.x==0) cycles[blockIdx.x]=end-start;
}

void codec_run(int64_t p, int64_t n, int64_t base, int64_t mode, bool registers,
               int64_t repeats, int64_t sums, int64_t cycles, int64_t blocks, int64_t stream) {
  const auto d=reinterpret_cast<const uint8_t*>(p);
  auto s=reinterpret_cast<unsigned long long*>(sums), c=reinterpret_cast<unsigned long long*>(cycles);
  auto st=reinterpret_cast<cudaStream_t>(stream);
  #define LAUNCH(M,R) probe<M,R><<<blocks,256,0,st>>>(d,n,base,repeats,s,c)
  if (registers) {
    if (mode==12) { LAUNCH(12,true); } else if (mode==11) { LAUNCH(11,true); } else { LAUNCH(0,true); }
  } else {
    if (mode==12) { LAUNCH(12,false); } else if (mode==11) { LAUNCH(11,false); } else { LAUNCH(0,false); }
  }
  const auto err=cudaGetLastError();
  if (err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
std::vector<int64_t> codec_resources(int64_t mode, bool registers) {
  cudaFuncAttributes a{};
  cudaError_t err;
  #define ATTR(M,R) cudaFuncGetAttributes(&a,probe<M,R>)
  if (registers) {
    if (mode==12) err=ATTR(12,true); else if (mode==11) err=ATTR(11,true); else err=ATTR(0,true);
  } else {
    if (mode==12) err=ATTR(12,false); else if (mode==11) err=ATTR(11,false); else err=ATTR(0,false);
  }
  if (err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  return {a.numRegs, int64_t(a.localSizeBytes), int64_t(a.sharedSizeBytes), a.maxThreadsPerBlock};
}
