"""Diagnostic nonsplit-K BF16 GEMM; not a submission.

Original full-K output tiling and two-CTA weight multicast experiment. Helpers,
SW128 layout and epilogue follow approved vjeux/alygg77 muse_bf16_proj.py
(Apache-2.0, vjeux f35a41e, alygg77 6cae49a). No external code imported.
All arithmetic is BF16 operands, FP32 WGMMA accumulation, BF16 final rounding.
CL2 pairs token tiles; each CTA computes its own complete K, no partial sums.
"""
import hashlib
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

# (token tile, independent weight-row warpgroups, stages, cluster CTAs).
# Matched plain/multicast pairs; only cluster policy differs within each pair.
CONFIGS = [(bn, wg, st, cl) for bn, wg, st in ((64,1,6), (64,2,6), (128,1,6), (128,2,5), (256,1,4), (32,1,8)) for cl in (1,2)]
CPP = r'''
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdint.h>
#include <vector>
void dense_launch(int,int64_t,int64_t,int64_t,int,int,int,int64_t);
std::vector<int64_t> dense_info(int);
'''
CUDA = r'''
// Portions Copyright vjeux / alygg77, Apache-2.0; see Python header.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>
#define D __device__ __forceinline__
D void init(uint32_t p,int count) { asm volatile("mbarrier.init.shared.b64 [%0], %1;"::"r"(p),"r"(count):"memory"); }
D void expect(uint32_t p,int bytes) { asm volatile("{ .reg .b64 s; mbarrier.arrive.expect_tx.shared.b64 s,[%0],%1; }"::"r"(p),"r"(bytes):"memory"); }
D void wait_bar(uint32_t p,int phase) {
  unsigned ok;
  do { asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2,0x989680; selp.u32 %0,1,0,p; }":"=r"(ok):"r"(p),"r"(phase):"memory"); } while(!ok);
}
D void tma(uint32_t dst,const CUtensorMap* map,uint32_t bar,int k,int row) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%3,%4}],[%2];"::"r"(dst),"l"(map),"r"(bar),"r"(k),"r"(row):"memory");
}
D void multi(uint32_t dst,const CUtensorMap* map,uint32_t bar,int k,int row) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%3,%4}],[%2],%5;"::"r"(dst),"l"(map),"r"(bar),"r"(k),"r"(row),"h"(uint16_t(3)):"memory");
}
@CONSUMERS@

template<int BN,int WG,int S,int CL>
__global__ __launch_bounds__(WG*128+32,1) void dense_kernel(
 const __grid_constant__ CUtensorMap mapW,const __grid_constant__ CUtensorMap mapX,
 __nv_bfloat16* out,int M,int N,int K) {
 constexpr int BM=64*WG, AB=BM*128, ST=(BM+BN)*128, NT=WG*128, NA=BN/2;
 extern __shared__ __align__(1024) unsigned char mem[];
 const uint32_t base=uint32_t(__cvta_generic_to_shared(mem));
 const uint32_t full=base+S*ST, emptyX=full+8*S, emptyW=emptyX+8*S;
 const int tid=threadIdx.x, warp=tid/32, lane=tid%32, wg=tid/128;
 const int rank=blockIdx.x%CL, job=blockIdx.x/CL;
 const int row=(job%(N/BM))*BM, token=(job/(N/BM))*BN*CL+rank*BN;
 const int steps=K/64;
 if(tid==0) {
   for(int s=0;s<S;++s) { init(full+8*s,1); init(emptyX+8*s,WG); init(emptyW+8*s,WG*CL); }
   asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 __syncthreads();
 if constexpr(CL==2) {
   asm volatile("barrier.cluster.arrive.aligned;":::"memory");
   asm volatile("barrier.cluster.wait.aligned;":::"memory");
 }
 uint32_t remoteW=emptyW;
 if constexpr(CL==2) asm volatile("mapa.shared::cluster.u32 %0,%1,0;":"=r"(remoteW):"r"(emptyW));
 if(warp==WG*4) {
   if(lane==0) {
     asm volatile("prefetch.tensormap [%0];"::"l"(&mapW):"memory");
     asm volatile("prefetch.tensormap [%0];"::"l"(&mapX):"memory");
     for(int i=0;i<steps;++i) {
       const int slot=i%S;
       if(i>=S) wait_bar((CL==2 && rank==0 ? emptyW : emptyX)+8*slot,((i/S)-1)&1);
       expect(full+8*slot,ST);
       if constexpr(CL==2) { if(rank==0) multi(base+slot*ST,&mapW,full+8*slot,i*64,row); }
       else tma(base+slot*ST,&mapW,full+8*slot,i*64,row);
       tma(base+slot*ST+AB,&mapX,full+8*slot,i*64,token);
     }
   }
 } else {
   float acc[NA];
   @CONSUME_CALL@
   // All consumer warpgroups must finish the ring before reusing it for stores.
   asm volatile("bar.sync 1,%0;"::"n"(NT):"memory");
   auto* stage=reinterpret_cast<__nv_bfloat16*>(mem);
   const int g=lane/4,t4=lane%4,r0=(warp%4)*16;
   #pragma unroll
   for(int j=0;j<BN/8;++j) {
     const int t=8*j+2*t4;
     auto* p=stage+(wg*BN+t)*64+r0+g;
     p[0]=__float2bfloat16_rn(acc[4*j]); p[8]=__float2bfloat16_rn(acc[4*j+2]);
     p[64]=__float2bfloat16_rn(acc[4*j+1]); p[72]=__float2bfloat16_rn(acc[4*j+3]);
   }
   asm volatile("bar.sync 2,%0;"::"n"(NT):"memory");
   for(int i=tid;i<WG*BN*8;i+=NT) {
     const int chunk=i%8,t=(i/8)%BN,w=i/(8*BN);
     if(token+t<M) *reinterpret_cast<uint4*>(out+(size_t)(token+t)*N+row+w*64+chunk*8)
       =*reinterpret_cast<const uint4*>(stage+(w*BN+t)*64+chunk*8);
   }
 }
 __syncthreads();
 if constexpr(CL==2) {
   asm volatile("barrier.cluster.arrive.aligned;":::"memory");
   asm volatile("barrier.cluster.wait.aligned;":::"memory");
 }
}
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
CUtensorMap map2d(void* ptr,int rows,int cols,int boxrows) {
 void* p=nullptr; cudaDriverEntryPointQueryResult query;
 ck(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&p,cudaEnableDefault,&query));
 if(!p) throw std::runtime_error("missing TensorMapEncodeTiled");
 auto encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
 CUtensorMap map; cuuint64_t dims[2]={uint64_t(cols),uint64_t(rows)},strides[1]={uint64_t(cols)*2};
 cuuint32_t box[2]={64,uint32_t(boxrows)},unit[2]={1,1};
 auto rc=encode(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,strides,box,unit,
 CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
 if(rc!=CUDA_SUCCESS) throw std::runtime_error("tensor map encode failed");
 return map;
}
template<int BN,int WG,int S,int CL>
std::vector<int64_t> info_t() {
 constexpr int smem=S*(64*WG+BN)*128+24*S;
 auto fn=dense_kernel<BN,WG,S,CL>;
 ck(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 cudaFuncAttributes a; ck(cudaFuncGetAttributes(&a,fn));
 int active; ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,WG*128+32,smem));
 return {smem,a.numRegs,int64_t(a.localSizeBytes),active};
}
template<int BN,int WG,int S,int CL>
void launch_t(int64_t wp,int64_t xp,int64_t op,int M,int N,int K,int64_t stream) {
 constexpr int BM=WG*64,smem=S*(BM+BN)*128+24*S;
 auto fn=dense_kernel<BN,WG,S,CL>;
 ck(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
 auto W=map2d(reinterpret_cast<void*>(wp),N,K,BM);
 auto X=map2d(reinterpret_cast<void*>(xp),M,K,BN);
 cudaLaunchConfig_t config{};
 config.gridDim=dim3((N/BM)*((M+BN*CL-1)/(BN*CL))*CL);
 config.blockDim=dim3(WG*128+32); config.dynamicSmemBytes=smem;
 config.stream=reinterpret_cast<cudaStream_t>(stream);
 cudaLaunchAttribute attr{};
 if constexpr(CL==2) { attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=2;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;config.attrs=&attr;config.numAttrs=1; }
 ck(cudaLaunchKernelEx(&config,fn,W,X,reinterpret_cast<__nv_bfloat16*>(op),M,N,K));
}
void dense_launch(int cfg,int64_t w,int64_t x,int64_t out,int m,int n,int k,int64_t stream) {
 @LAUNCHES@
 throw std::runtime_error("unknown config");
}
std::vector<int64_t> dense_info(int cfg) {
 @INFOS@
 throw std::runtime_error("unknown config");
}
'''


def consumer(bn):
    na = bn//2
    regs = ','.join(f'%{q}' for q in range(na))
    # A single compiler-opaque consumer loop prevents ptxas C7514 serialization.
    # Parameters: base,wg_offset,full,emptyX,emptyW,lane_in_wg,steps,S,ST,AB,CL.
    p = [f'%{na+i}' for i in range(11)]
    ops = ['{', '.reg .u32 i,slot,phase,ptr,tmp,prev,wa,xb;',
           '.reg .b64 da,db,ds,unused;', '.reg .pred ready,more,leader,multi;',
           *[f'mov.f32 %{q},0f00000000;' for q in range(na)],
           'wgmma.fence.sync.aligned;', 'mov.u32 i,0;', 'consume_loop:',
           f'rem.u32 slot,i,{p[7]};', f'div.u32 phase,i,{p[7]};', 'and.b32 phase,phase,1;',
           f'mad.lo.u32 ptr,slot,8,{p[2]};', 'ready_loop:',
           'mbarrier.try_wait.parity.shared::cta.b64 ready,[ptr],phase,0x989680;',
           '@!ready bra ready_loop;', f'mad.lo.u32 wa,slot,{p[8]},{p[0]};',
           f'add.u32 xb,wa,{p[9]};', f'add.u32 wa,wa,{p[1]};',
           'shr.u32 wa,wa,4;', 'shr.u32 xb,xb,4;', 'cvt.u64.u32 da,wa;', 'cvt.u64.u32 db,xb;',
           'or.b64 da,da,0x4000004000010000;', 'or.b64 db,db,0x4000004000010000;']
    ops += ['wgmma.fence.sync.aligned;']
    for j in range(4):
        ops += [f'wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 {{{regs}}},da,db,1,1,1,0,0;']
        if j < 3:
            ops += ['add.u64 da,da,2;', 'add.u64 db,db,2;']
    ops += ['wgmma.commit_group.sync.aligned;', 'wgmma.wait_group.sync.aligned 1;',
            'setp.gt.u32 more,i,0;', f'setp.eq.u32 leader,{p[5]},0;', 'and.pred leader,leader,more;',
            '@!leader bra no_release;', 'sub.u32 prev,i,1;', f'rem.u32 prev,prev,{p[7]};',
            f'mad.lo.u32 ptr,prev,8,{p[3]};', 'mbarrier.arrive.shared.b64 unused,[ptr];',
            f'setp.eq.u32 multi,{p[10]},2;', '@!multi bra no_release;',
            f'mad.lo.u32 ptr,prev,8,{p[4]};', 'mbarrier.arrive.release.cluster.shared::cluster.b64 _,[ptr];',
            'no_release:', 'add.u32 i,i,1;', f'setp.lt.u32 more,i,{p[6]};', '@more bra consume_loop;',
            'wgmma.wait_group.sync.aligned 0;', '}']
    text = '\\n'.join(ops)
    outputs = ','.join(f'"=f"(acc[{i}])' for i in range(na))
    return f'''template<int S,int ST,int AB,int CL> D void consume{bn}(float* acc,uint32_t base,uint32_t off,uint32_t full,uint32_t ex,uint32_t ew,int tid,int steps) {{
 asm volatile("{text}":{outputs}:"r"(base),"r"(off),"r"(full),"r"(ex),"r"(ew),"r"(tid),"r"(steps),"n"(S),"n"(ST),"n"(AB),"n"(CL):"memory");
}}'''


def source():
    widths = sorted({c[0] for c in CONFIGS})
    return (CUDA.replace('@CONSUMERS@', '\n'.join(consumer(bn) for bn in widths))
            .replace('@CONSUME_CALL@', ' '.join(f'if constexpr(BN=={bn}) consume{bn}<S,ST,AB,CL>(acc,base,wg*8192,full,emptyX,remoteW,tid%128,steps);' for bn in widths))
            .replace('@LAUNCHES@', '\n'.join(f'if(cfg=={i}) {{launch_t<{",".join(map(str,c))}>(w,x,out,m,n,k,stream);return;}}' for i,c in enumerate(CONFIGS)))
            .replace('@INFOS@', '\n'.join(f'if(cfg=={i}) return info_t<{",".join(map(str,c))}>();' for i,c in enumerate(CONFIGS))))


def build():
    text = source()
    Path('dense_tma_generated.cu').write_text(text)
    digest = hashlib.sha256((CPP+text).encode()).hexdigest()[:16]
    return load_inline('dense_tma_'+digest, CPP, cuda_sources=text,
                       functions=['dense_launch','dense_info'], no_implicit_headers=True,
                       with_pytorch_error_handling=False, verbose=True,
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','-Xptxas=-v'])


def run(ext, cfg, weight, x):
    m,k=x.shape
    n=weight.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    ext.dense_launch(cfg,weight.data_ptr(),x.data_ptr(),out.data_ptr(),m,n,k,torch.cuda.current_stream().cuda_stream)
    return out
