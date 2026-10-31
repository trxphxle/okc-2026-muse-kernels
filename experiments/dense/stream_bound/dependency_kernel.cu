// SPDX-License-Identifier: Apache-2.0
// Diagnostic byte ceiling, NOT a production layer or a submission.
// TMA/mbarrier patterns: Copyright vjeux (f35a41e), Apache-2.0, and
// Copyright 2026 alygg77 (6cae49a), Apache-2.0. Glue operation ordering is
// derived from the approved repaired decode_stream_base.py (_SL_GLUE_CUDA).
// dependency_kernel.py prepends dependency_abi.cpp_enum(): no copied indices.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace dense_dependency {
using BF = __nv_bfloat16;
constexpr int Threads=256, Slots=4, StageBytes=32768, Dynamic=Slots*StageBytes;
constexpr uint64_t DelayNs=20000;
constexpr unsigned FullWarp=0xffffffffu;

struct Shared {
  int64_t params[FLAT_COUNT+1];
  alignas(8) uint64_t bars[Slots];
  uint64_t source[Slots];
  int sizes[Slots];
  uint32_t parity[Slots];
  uint32_t epoch;
  uint32_t descriptor_errors;
  float sums[64];
};

template<class U> __device__ __forceinline__ U* ptr(const int64_t* p,int field) {
  return reinterpret_cast<U*>(p[field]);
}
__device__ __forceinline__ uint32_t address(const void* p) {
  return uint32_t(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint64_t timer() {
  uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory"); return t;
}
__device__ __forceinline__ uint32_t acquire(const uint32_t* p) {
  uint32_t v; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory"); return v;
}
__device__ __forceinline__ void release(uint32_t* p,uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");
}
__device__ __forceinline__ void error(const int64_t* p,int bit) {
  atomicOr(reinterpret_cast<unsigned int*>(ptr<int>(p,ERROR)),unsigned(bit));
}
// Each producer releases its own cache-line-padded epoch. A warp acquires all
// required producers and joins before lane0 publishes the global readiness.
// This avoids a serialized GPU acq_rel RMW chain and needs no replay reset.
__device__ __forceinline__ void arrive(const int64_t* p,int flag,int expected,
                                      uint32_t epoch,bool validation) {
  if(threadIdx.x>=32 || blockIdx.x>=unsigned(expected)) return;
  const int lane=threadIdx.x;
  auto arrivals=ptr<uint32_t>(p,ARRIVALS)+(int64_t(flag)*gridDim.x)*32;
  if(lane==0) {
    auto own=arrivals+int64_t(blockIdx.x)*32;
    if(validation && acquire(own)==epoch) error(p,8);
    release(own,epoch);
  }
  __syncwarp();
  if(blockIdx.x==0) {
    for(;;) {
      bool ready=true;
      for(int producer=lane;producer<expected;producer+=32)
        ready=ready && acquire(arrivals+int64_t(producer)*32)==epoch;
      if(__all_sync(FullWarp,ready)) break;
      __nanosleep(32);
    }
    // Warp synchronization transfers every lane's acquire to the publishing lane.
    __syncwarp();
    if(lane==0) {
      if(validation && p[DELAY_STAGE]==flag) {
        const uint64_t begin=timer();
        while(timer()-begin<DelayNs) __nanosleep(32);
      }
      if(validation) ptr<int64_t>(p,READY_TIMES)[flag]=int64_t(timer());
      release(ptr<uint32_t>(p,FLAGS)+flag,epoch);
    }
  }
  __syncwarp();
}
// Warp 0 only. The warp acquire is propagated to helper warps by the CTA
// barriers before new-KV, B1, route and B2 work, not by a grid barrier.
__device__ __forceinline__ void gate(const int64_t* p,int flag,uint32_t epoch) {
  if(threadIdx.x==0) {
    for(;;) {
      const uint32_t value=acquire(ptr<uint32_t>(p,FLAGS)+flag);
      if(value==epoch) break;
      if(int32_t(value-epoch)>0) { error(p,16); break; }
    }
  }
  __syncwarp();
}
__device__ __forceinline__ void wait(uint32_t bar,uint32_t parity) {
  asm volatile("{ .reg .pred done; again_%=: mbarrier.try_wait.parity.shared::cta.b64 done,[%0],%1; @!done bra again_%=; }"
               ::"r"(bar),"r"(parity):"memory");
}
__device__ __forceinline__ float add(float x,float y) { return __fadd_rn(x,y); }
__device__ __forceinline__ float mul(float x,float y) { return __fmul_rn(x,y); }
__device__ __forceinline__ float fma(float x,float y,float z) { return __fmaf_rn(x,y,z); }
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float invnorm(float sum,float scale,float eps) {
  float v=fma(sum,scale,eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(v):"f"(v)); return v;
}
__device__ __forceinline__ float butterfly(float v,int first=16) {
  for(int d=first;d;d>>=1) v=add(v,__shfl_xor_sync(FullWarp,v,d));
  return v;
}
template<int N> __device__ __forceinline__ float squares(const float (&v)[N]) {
  float q[N/2];
  #pragma unroll
  for(int j=0;j<N/2;++j) q[j]=fma(v[2*j],v[2*j],mul(v[2*j+1],v[2*j+1]));
  #pragma unroll
  for(int n=N/2;n>1;n/=2) {
    #pragma unroll
    for(int j=0;j<n/2;++j) q[j]=add(q[2*j],q[2*j+1]);
  }
  return q[0];
}
__device__ __forceinline__ float block_sum(float v,float* sums) {
  v=butterfly(v);
  if((threadIdx.x&31)==0) sums[threadIdx.x/32]=v;
  __syncthreads();
  v=sums[threadIdx.x&7];
  v=butterfly(v,4);
  __syncthreads(); // no helper may overwrite scratch while a peer still reads
  return v;
}

struct View {
  const int64_t* fixed;
  const int64_t* bank;
  int64_t first, count;
  int per;
  uint32_t mask;
  __device__ __forceinline__ int64_t size() const {
    return count+int64_t(__popc(mask))*per;
  }
  __device__ __forceinline__ const int64_t* item(int64_t i) const {
    if(i<count) return fixed+3*(first+i);
    i-=count;
    int rank=int(i/per);
    uint32_t remaining=mask;
    while(rank--) remaining&=remaining-1;
    const int expert=__ffs(remaining)-1;
    return bank+3*(int64_t(expert)*per+i%per);
  }
};
__device__ __forceinline__ View view(const int64_t* p,int phase,bool flat=false) {
  if(flat) return {ptr<const int64_t>(p,FLAT_CHUNKS),nullptr,0,p[FLAT_COUNT],0,0};
  const auto range=ptr<const int64_t>(p,RANGES)+2*phase;
  const uint32_t mask=phase==6 ? *ptr<const uint32_t>(p,ACTIVE_MASK) : 0;
  return {ptr<const int64_t>(p,CHUNKS),ptr<const int64_t>(p,EXP_CHUNKS),
          range[0],range[1],int(p[EXP_CHUNKS_PER]),mask};
}
__device__ __forceinline__ int64_t local_count(const View& v) {
  const int64_t n=v.size();
  return n>blockIdx.x ? 1+(n-1-blockIdx.x)/gridDim.x : 0;
}
__device__ __forceinline__ int64_t* stamp(const int64_t* p,int phase) {
  return ptr<int64_t>(p,STAMPS)+(int64_t(blockIdx.x)*9+phase)*3;
}
__device__ __forceinline__ void issue(const int64_t* p,const View& v,int64_t local,
    int slot,unsigned char* data,Shared& s,bool validation) {
  if(threadIdx.x==0) {
    const auto item=v.item(blockIdx.x+local*gridDim.x);
    const uint64_t source=uint64_t(item[0]);
    const int64_t bytes=item[1];
    s.source[slot]=source;
    if(validation && (!source || (source&15) || bytes<=0 || bytes>StageBytes || (bytes&15) || item[2]!=0)) {
      s.sizes[slot]=0; s.descriptor_errors|=2;
    } else {
      s.sizes[slot]=int(bytes);
      const uint32_t b=address(s.bars+slot),d=address(data+slot*StageBytes);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(b),"r"(uint32_t(bytes)):"memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"
                   ::"r"(d),"l"(source),"r"(uint32_t(bytes)),"r"(b):"memory");
    }
  }
  __syncwarp();
}
// The descriptor contains complete K rows; only those row outputs become live.
// No oracle row is copied by a prefetch issue, or before its consumer's gate.
__device__ __forceinline__ void copy_rows(const int64_t* p,int phase,uint64_t source,int bytes) {
  int base_field, oracle_field, live_field, row_bytes, width;
  if(phase==2) {
    base_field=O_W; oracle_field=O_ORACLE; live_field=O_LIVE; row_bytes=2048; width=8192;
  } else if(phase==7) {
    base_field=UN_W; oracle_field=UN_ORACLE; live_field=UN_LIVE; row_bytes=8192; width=1024;
  } else if(phase==4) {
    if(source>=uint64_t(p[POOL_W]) && source<uint64_t(p[POOL_W])+4096ull*16384) {
      base_field=POOL_W; oracle_field=POOLED_ORACLE; live_field=POOLED_LIVE; row_bytes=16384; width=4096;
    } else {
      base_field=ROUTER_W; oracle_field=SCORES_ORACLE; live_field=SCORES_LIVE; row_bytes=32768; width=256;
    }
  } else return;
  const uint64_t base=uint64_t(p[base_field]);
  if(source<base || source+bytes>base+uint64_t(width)*row_bytes ||
     (source-base)%row_bytes || bytes%row_bytes) {
    if(threadIdx.x==0) error(p,4);
    return;
  }
  const int first=int((source-base)/row_bytes), rows=bytes/row_bytes;
  for(int i=threadIdx.x;i<rows*int(p[T]);i+=32) {
    const int offset=(i/rows)*width+first+i%rows;
    if(live_field==SCORES_LIVE)
      ptr<float>(p,live_field)[offset]=ptr<const float>(p,oracle_field)[offset];
    else ptr<BF>(p,live_field)[offset]=ptr<const BF>(p,oracle_field)[offset];
  }
}
__device__ __forceinline__ void prefetch(const int64_t* p,int phase,unsigned char* data,
                                        Shared& s,bool validation) {
  const View v=view(p,phase);
  const int64_t n=local_count(v);
  if(validation && threadIdx.x==0 && n) stamp(p,phase)[0]=int64_t(timer());
  #pragma unroll
  for(int slot=0;slot<2;++slot) if(slot<n) issue(p,v,slot,slot,data,s,validation);
}
// Warp 0 exclusively drives the ring. Protected slots 0/1 are retained until
// the future operation's gate; intervening phases use only slots 2/3. At the
// consumer, the held chunks join a four-slot ring without discard/reload.
__device__ __forceinline__ void stream_phase(const int64_t* p,int phase,unsigned char* data,
    Shared& s,bool validation,int slot_base=0,int slots=4,bool prefetched=false,bool flat=false) {
  const View v=view(p,phase,flat);
  const int64_t n=local_count(v);
  if(validation && threadIdx.x==0) {
    stamp(p,phase)[2]=int64_t(timer());
    if(n && !prefetched) stamp(p,phase)[0]=int64_t(timer());
  }
  #pragma unroll
  for(int j=0;j<4;++j)
    if(j<slots && j<n && !(prefetched && j<2)) issue(p,v,j,slot_base+j,data,s,validation);
  uint64_t bytes=0;
  uint32_t checksum=0;
  for(int64_t i=0;i<n;++i) {
    const int slot=slot_base+int(i%slots);
    const int size=s.sizes[slot];
    if(size) {
      wait(address(s.bars+slot),s.parity[slot]);
      if(validation) {
        if(threadIdx.x==0 && i+1==n) stamp(p,phase)[1]=int64_t(timer());
        const volatile uint32_t* words=reinterpret_cast<const volatile uint32_t*>(data+slot*StageBytes);
        for(int j=threadIdx.x;j<size/4;j+=32) checksum^=words[j];
      }
      if(!flat) copy_rows(p,phase,s.source[slot],size);
      if(threadIdx.x==0) bytes+=uint64_t(size);
      __syncwarp(); // every shared reader/copy finishes before refill
      if(threadIdx.x==0) s.parity[slot]^=1;
      __syncwarp();
    }
    if(i+slots<n) issue(p,v,i+slots,slot,data,s,validation);
  }
  if(validation) {
    #pragma unroll
    for(int d=16;d;d/=2) checksum^=__shfl_xor_sync(FullWarp,checksum,d);
  }
  if(threadIdx.x==0) {
    auto sink=ptr<int64_t>(p,SINK)+(int64_t(blockIdx.x)*9+phase)*2;
    sink[0]=int64_t(bytes); sink[1]=int64_t(checksum);
  }
  __syncwarp(); // release arrival includes every oracle-output store
}

// Four original logical threads per physical thread reproduce the approved
// 1024-thread, V=4, two-repeat B1 reduction tree without a cluster. Register
// residency/spills must still be measured by the root's H200 build.
__device__ __forceinline__ int b1_column(int group,int j) {
  return (int(threadIdx.x)+group*256)*4+(j/4)*4096+j%4;
}
__device__ __forceinline__ float b1_sum(const float (&x)[32],float* sums) {
  #pragma unroll
  for(int group=0;group<4;++group) {
    float v[8];
    #pragma unroll
    for(int j=0;j<8;++j) v[j]=x[group*8+j];
    const float sum=butterfly(squares(v));
    if((threadIdx.x&31)==0) sums[group*8+threadIdx.x/32]=sum;
  }
  __syncthreads();
  const float result=butterfly(sums[threadIdx.x&31]);
  __syncthreads();
  return result;
}
__device__ __forceinline__ float2 b1_pair(const float (&x)[32],const float (&y)[32],float* sums) {
  #pragma unroll
  for(int group=0;group<4;++group) {
    float vx[8],vy[8];
    #pragma unroll
    for(int j=0;j<8;++j) { vx[j]=x[group*8+j]; vy[j]=y[group*8+j]; }
    const float sx=butterfly(squares(vx)),sy=butterfly(squares(vy));
    if((threadIdx.x&31)==0) {
      const int offset=group*8+threadIdx.x/32;
      sums[offset]=sx; sums[offset+32]=sy;
    }
  }
  __syncthreads();
  const float2 result=make_float2(butterfly(sums[threadIdx.x&31]),
                                 butterfly(sums[32+(threadIdx.x&31)]));
  __syncthreads();
  return result;
}
__device__ __forceinline__ void boundary1(const int64_t* p,Shared& s) {
  const int row=blockIdx.x;
  float residual[32],carry[32];
  #pragma unroll
  for(int group=0;group<4;++group) {
    #pragma unroll
    for(int j=0;j<8;++j) {
      const int c=b1_column(group,j);
      residual[group*8+j]=__bfloat162float(ptr<const BF>(p,O_LIVE)[row*8192+c]);
    }
  }
  const float r1=invnorm(b1_sum(residual,s.sums),1.f/8192,1.e-8f);
  #pragma unroll
  for(int group=0;group<4;++group) {
    #pragma unroll
    for(int j=0;j<8;++j) {
      const int k=group*8+j,c=b1_column(group,j);
      const float n=bf(mul(residual[k],r1));
      const float a=ptr<const float>(p,ALPHA1)[c],b=ptr<const float>(p,BETA1)[c];
      const float stream=ptr<const float>(p,STREAM0)[row*8192+c];
      residual[k]=(p[T]>1 && c<4096) ? fma(a,stream,mul(b,n)) : fma(b,n,mul(a,stream));
      const float value=mul(ptr<const float>(p,ALPHA2)[c],residual[k]);
      ptr<float>(p,CARRY)[row*8192+c]=value;
      carry[k]=c>=1024 ? value : 0.f;
    }
  }
  const float2 sums=b1_pair(residual,carry,s.sums);
  const float r2=invnorm(sums.x,1.f/8192,1.e-5f);
  if(threadIdx.x==0) ptr<float>(p,TAIL)[row]=sums.y;
  #pragma unroll
  for(int group=0;group<4;++group) {
    #pragma unroll
    for(int j=0;j<8;++j) {
      const int c=b1_column(group,j);
      const float h=mul(mul(residual[group*8+j],r2),__bfloat162float(ptr<const BF>(p,PRE1)[c]));
      ptr<BF>(p,H_LIVE)[row*8192+c]=__float2bfloat16_rn(h);
      ptr<float>(p,HR_LIVE)[row*8192+c]=h;
    }
  }
}
__device__ __forceinline__ void route(const int64_t* p,Shared& s,bool validation) {
  const int row=blockIdx.x,tid=threadIdx.x;
  float pooled[16];
  #pragma unroll
  for(int j=0;j<16;++j) pooled[j]=__bfloat162float(ptr<const BF>(p,POOLED_LIVE)[row*4096+tid+j*256]);
  const float inv=invnorm(block_sum(squares(pooled),s.sums),1.f/4096,1.e-5f);
  #pragma unroll
  for(int j=0;j<16;++j) {
    const int c=tid+j*256;
    const float x=mul(mul(pooled[j],inv),__bfloat162float(ptr<const BF>(p,GPREXP)[c]));
    ptr<BF>(p,XM_LIVE)[row*4096+c]=__float2bfloat16_rn(x);
  }
  if(tid<32) {
    unsigned long long keys[8];
    #pragma unroll
    for(int j=0;j<8;++j) {
      const int expert=tid+j*32;
      float score=ptr<const float>(p,SCORES_LIVE)[row*256+expert];
      if(validation && !isfinite(score)) error(p,32);
      if(score==0.f) score=0.f; // signed zeros tie by the lower expert ID
      const uint32_t bits=__float_as_uint(score);
      const uint32_t ordered=(bits&0x80000000u) ? ~bits : bits^0x80000000u;
      keys[j]=(uint64_t(ordered)<<32)|uint32_t(255-expert);
    }
    uint32_t selected=0;
    const int local_start=*ptr<const int>(p,LOCAL_START);
    if(tid==0 && (local_start<0 || local_start>=256)) error(p,64);
    #pragma unroll
    for(int k=0;k<8;++k) {
      unsigned long long best=0;
      #pragma unroll
      for(int j=0;j<8;++j) best=keys[j]>best ? keys[j] : best;
      #pragma unroll
      for(int d=16;d;d/=2) {
        const auto other=__shfl_xor_sync(FullWarp,best,d);
        best=other>best ? other : best;
      }
      const int expert=255-int(best&255);
      if(tid==0) {
        ptr<int>(p,IDS_OUT)[row*8+k]=expert;
        const int local=(expert-local_start+256)&255;
        if(local<32) selected|=uint32_t(1)<<local;
      }
      #pragma unroll
      for(int j=0;j<8;++j) if(keys[j]==best) keys[j]=0;
    }
    if(tid==0) atomicOr(ptr<uint32_t>(p,ACTIVE_MASK),selected);
  }
}
// Exact repaired sliced formula: every tile recomputes the 1024-value branch
// norm and head norm; tail is the legally precomputed B1 carry squared sum.
__device__ __forceinline__ void boundary2(const int64_t* p,Shared& s) {
  const int row=blockIdx.x/8,tile=blockIdx.x%8,tid=threadIdx.x;
  float head[4];
  #pragma unroll
  for(int j=0;j<4;++j) {
    const int c=tid+j*256;
    head[j]=bf(mul(__bfloat162float(ptr<const BF>(p,UN_LIVE)[row*1024+c]),
                   __bfloat162float(ptr<const BF>(p,POST2)[c])));
  }
  const float inv=invnorm(block_sum(squares(head),s.sums),1.f/8192,1.e-8f);
  #pragma unroll
  for(int j=0;j<4;++j) {
    const int c=tid+j*256;
    const float n=bf(mul(head[j],inv));
    head[j]=fma(ptr<const float>(p,BETA2)[c],n,ptr<const float>(p,CARRY)[row*8192+c]);
  }
  const float sum=block_sum(squares(head),s.sums);
  const float invnext=invnorm(add(ptr<const float>(p,TAIL)[row],sum),1.f/8192,1.e-5f);
  #pragma unroll
  for(int j=0;j<4;++j) {
    const int c=tile*1024+tid+j*256;
    const float value=tile==0 ? head[j] : ptr<const float>(p,CARRY)[row*8192+c];
    ptr<float>(p,SOUT)[row*8192+c]=value;
    ptr<BF>(p,OUT)[row*8192+c]=__float2bfloat16_rn(mul(mul(value,invnext),__bfloat162float(ptr<const BF>(p,PRE2)[c])));
  }
}

__global__ __launch_bounds__(Threads,1) void dependency_stream(const int64_t* input_params,int mode) {
  extern __shared__ __align__(128) unsigned char data[];
  __shared__ Shared s;
  if(threadIdx.x<=FLAT_COUNT) s.params[threadIdx.x]=input_params[threadIdx.x];
  __syncthreads();
  const int64_t* p=s.params;
  const bool validation=mode==2 || mode==3;
  const bool ahead=mode==1 || mode==3;
  const int tid=threadIdx.x,block=blockIdx.x,tokens=int(p[T]);
  if(tid==0) {
    if(block==0) {
      *ptr<int>(p,ERROR)=0;
      if(mode!=4) *ptr<uint32_t>(p,ACTIVE_MASK)=0;
    }
    // All CTAs capture the same epoch before phase0: its all-CTA arrival is
    // necessary even for an empty local phase0 slice, before QKV can publish.
    s.epoch=mode==4 ? 0 : acquire(ptr<uint32_t>(p,FLAGS)+QKV_DONE)+1;
    s.descriptor_errors=0;
    for(int phase=0;phase<9;++phase) {
      auto sink=ptr<int64_t>(p,SINK)+(int64_t(block)*9+phase)*2;
      sink[0]=0; sink[1]=0;
      if(validation) {
        auto ts=stamp(p,phase); ts[0]=0; ts[1]=0; ts[2]=0;
      }
    }
    #pragma unroll
    for(int slot=0;slot<Slots;++slot) {
      s.parity[slot]=0; s.sizes[slot]=0;
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(address(s.bars+slot)):"memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  if(p[BLOCKS]!=gridDim.x || tokens<1 || tokens>8 || p[EXP_CHUNKS_PER]<=0 || p[FLAT_COUNT]<0) {
    if(block==0 && tid==0) error(p,1);
    return;
  }
  if(mode==4) {
    if(tid<32) stream_phase(p,0,data,s,false,0,4,false,true);
    return;
  }
  const uint32_t epoch=s.epoch;
  if(tid<32) {
    stream_phase(p,0,data,s,validation);
    arrive(p,QKV_DONE,gridDim.x,epoch,validation);
    if(ahead) prefetch(p,2,data,s,validation);
    gate(p,QKV_DONE,epoch);
  }
  __syncthreads();
  if(block<tokens) {
    const int which=tid/128,column=tid%128;
    const int64_t row=ptr<const int64_t>(p,CACHE_LOC)[block];
    ptr<unsigned char>(p,which ? V_CACHE : K_CACHE)[row*128+column]=
      ptr<const unsigned char>(p,NEW_KV)[(which*tokens+block)*128+column];
    // Generic global stores must be visible to the later async-proxy TMA read.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __syncthreads();
    arrive(p,KV_WRITTEN,tokens,epoch,validation);
  }
  if(tid<32) {
    gate(p,KV_WRITTEN,epoch);
    stream_phase(p,1,data,s,validation,ahead ? 2 : 0,ahead ? 2 : 4);
    arrive(p,ATTN_DONE,gridDim.x,epoch,validation);
    gate(p,ATTN_DONE,epoch);
    stream_phase(p,2,data,s,validation,0,4,ahead);
    if(ahead) prefetch(p,4,data,s,validation);
    arrive(p,O_DONE,gridDim.x,epoch,validation);
    gate(p,O_DONE,epoch);
    stream_phase(p,3,data,s,validation,ahead ? 2 : 0,ahead ? 2 : 4);
  }
  __syncthreads();
  if(block<tokens) boundary1(p,s);
  __syncthreads();
  if(tid<32) {
    arrive(p,B1_DONE,tokens,epoch,validation);
    gate(p,B1_DONE,epoch);
    stream_phase(p,4,data,s,validation,0,4,ahead);
    if(ahead) prefetch(p,7,data,s,validation);
    arrive(p,RP_DONE,gridDim.x,epoch,validation);
    gate(p,RP_DONE,epoch);
    stream_phase(p,5,data,s,validation,ahead ? 2 : 0,ahead ? 2 : 4);
  }
  __syncthreads();
  if(block<tokens) route(p,s,validation);
  __syncthreads();
  if(tid<32) {
    arrive(p,ROUTES_READY,tokens,epoch,validation);
    gate(p,ROUTES_READY,epoch);
    stream_phase(p,6,data,s,validation,ahead ? 2 : 0,ahead ? 2 : 4);
    arrive(p,EXPERTS_DONE,gridDim.x,epoch,validation);
    gate(p,EXPERTS_DONE,epoch);
    stream_phase(p,7,data,s,validation,0,4,ahead);
    arrive(p,UNPOOL_DONE,gridDim.x,epoch,validation);
    gate(p,UNPOOL_DONE,epoch);
    stream_phase(p,8,data,s,validation);
  }
  __syncthreads();
  if(block<tokens*8) boundary2(p,s);
  __syncthreads();
  // Early phase0/prefetch metadata failures must not race block0's ERROR
  // initialization. By this point QKV's join guarantees startup has finished.
  if(tid==0 && s.descriptor_errors) error(p,int(s.descriptor_errors));
  arrive(p,B2_DONE,tokens*8,epoch,validation);
}

void check(cudaError_t status) {
  if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct LaunchGeometry { int device=-1; int blocks=0; };
thread_local LaunchGeometry geometry;
} // namespace dense_dependency

std::vector<int64_t> dep_info() {
  using namespace dense_dependency;
  int device; check(cudaGetDevice(&device));
  cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties,device));
  if(properties.major!=9 || properties.minor!=0 || !properties.cooperativeLaunch)
    throw std::runtime_error("dependency ceiling requires an sm90a cooperative-launch device");
  check(cudaFuncSetAttribute(dependency_stream,cudaFuncAttributeMaxDynamicSharedMemorySize,Dynamic));
  cudaFuncAttributes attributes{}; check(cudaFuncGetAttributes(&attributes,dependency_stream));
  int active; check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dependency_stream,Threads,Dynamic));
  if(active!=1) throw std::runtime_error("dependency ceiling requires exactly one resident CTA per SM");
  geometry={device,properties.multiProcessorCount};
  return {attributes.numRegs,int64_t(attributes.sharedSizeBytes),int64_t(attributes.localSizeBytes),
          attributes.maxThreadsPerBlock,Dynamic,Threads,active,properties.multiProcessorCount,
          StageBytes,Slots,properties.cooperativeLaunch,geometry.blocks,FLAT_COUNT+1,int64_t(DelayNs)};
}

void dep_launch(int64_t param_ptr,int mode,int64_t stream) {
  using namespace dense_dependency;
  if(!param_ptr || (param_ptr&7) || mode<0 || mode>4)
    throw std::invalid_argument("invalid dependency stream arguments");
  int device; check(cudaGetDevice(&device));
  if(geometry.device!=device || geometry.blocks<=0)
    throw std::runtime_error("call dep_info on this thread/current device before capture");
  auto parameters=reinterpret_cast<const int64_t*>(param_ptr);
  void* args[]={&parameters,&mode};
  check(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(dependency_stream),
        dim3(geometry.blocks),dim3(Threads),args,Dynamic,reinterpret_cast<cudaStream_t>(stream)));
}
