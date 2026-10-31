"""Diagnostic: exact BF16 Huffman-exponent streams,128 values per addressable block.
Not a submission. New lane code; uses screen_dense.py's captured-event timer.
Low plane is sign+mantissa8; high plane codes the full exponent, with no escapes.
32-bit word offsets plus per-block word alignment support independent threads.
GPU reconstructs every BF16 bit for validation; timed paths retain only checksums.
Fixed12 comparison has the identical low layout and consumer loop, but a4-bit
exponent window with clipped tails. Its omitted corrections favor the control.
"""
import argparse
import hashlib
import heapq
import json
import statistics
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
import screen_dense as screen
import reference

CPP='''#include <pybind11/pybind11.h>
#include <stdint.h>
void decode(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int,int,bool,int64_t,int64_t,int64_t);
'''
CUDA=r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
template<bool Huff,bool Store>
__global__ void decoder(const unsigned char* low,const unsigned* stream,const unsigned* offsets,
 const unsigned* table,const int* tree,unsigned* sums,unsigned short* out,int groups,int base) {
 __shared__ unsigned lut[256]; __shared__ int nodes[512];
 for(int i=threadIdx.x;i<256;i+=blockDim.x) lut[i]=table[i];
 for(int i=threadIdx.x;i<512;i+=blockDim.x) nodes[i]=tree[i];
 __syncthreads();
 const int tid=blockIdx.x*blockDim.x+threadIdx.x;
 unsigned sum=0;
 for(int g=tid;g<groups;g+=gridDim.x*blockDim.x) {
   unsigned bit=0;
   const unsigned start=Huff?offsets[g]:0;
   for(int v=0;v<128;++v) {
     unsigned ex;
     if constexpr(Huff) {
       const unsigned word=bit>>5, shift=bit&31;
       const unsigned bits=__funnelshift_r(__ldg(stream+start+word),__ldg(stream+start+word+1),shift);
       const unsigned entry=lut[bits&255];
       unsigned length=entry&31;
       if(entry>>31) {
         int node=int(entry&0x7fffffffu);length=8;
         do { node=nodes[2*node+((bits>>length)&1)]; ++length; } while(node>=0);
         ex=unsigned(-node-1);
       } else ex=entry>>5;
       bit+=length;
     } else {
       const auto* codes=reinterpret_cast<const unsigned char*>(stream);
       ex=base+((__ldg(codes+(v/2)*groups+g)>>(4*(v%2)))&15);
     }
     const unsigned b=__ldg(low+v*groups+g);
     const unsigned value=(b&127)|((b&128)<<8)|(ex<<7);
     sum+=value;
     if constexpr(Store) out[g*128+v]=value;
   }
 }
 sums[tid]=sum;
}
void decode(int64_t low,int64_t stream,int64_t offsets,int64_t table,int64_t tree,int64_t sums,int64_t out,
 int groups,int base,bool store,int64_t mode,int64_t blocks,int64_t st) {
 const auto l=reinterpret_cast<const unsigned char*>(low);const auto s=reinterpret_cast<const unsigned*>(stream);
 const auto off=reinterpret_cast<const unsigned*>(offsets);const auto tab=reinterpret_cast<const unsigned*>(table);
 const auto tr=reinterpret_cast<const int*>(tree);auto sum=reinterpret_cast<unsigned*>(sums);auto dst=reinterpret_cast<unsigned short*>(out);
 auto stream_=reinterpret_cast<cudaStream_t>(st);
 #define RUN(H,S) decoder<H,S><<<blocks,256,0,stream_>>>(l,s,off,tab,tr,sum,dst,groups,base)
 if(mode==1) { if(store) { RUN(true,true); } else { RUN(true,false); } }
 else { if(store) { RUN(false,true); } else { RUN(false,false); } }
 const auto e=cudaGetLastError();if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
'''


def codes_for(hist):
    heap=[(int(count),i,i) for i,count in enumerate(hist) if count]
    heapq.heapify(heap);serial=256
    while len(heap)>1:
        a=heapq.heappop(heap);b=heapq.heappop(heap)
        heapq.heappush(heap,(a[0]+b[0],serial,(a[2],b[2])));serial+=1
    codes=np.zeros(256,dtype=np.uint32);lengths=np.zeros(256,dtype=np.int64)
    children=[]
    def walk(node,code=0,length=0):
        if isinstance(node,int):
            codes[node]=code;lengths[node]=max(1,length)
            return -node-1
        index=len(children);children.append([0,0])
        left=walk(node[0],code,length+1);right=walk(node[1],code|(1<<length),length+1)
        children[index]=[left,right]
        return index
    walk(heap[0][2])
    if max(lengths)>31:
        raise RuntimeError('Measured histogram needs a longer lookup window')
    table=np.zeros(256,dtype=np.uint32)
    tree=np.zeros((256,2),dtype=np.int32)
    tree[:len(children)]=children
    for bits in range(256):
        node=0;used=0
        while node>=0 and used<8:
            node=int(tree[node,(bits>>used)&1]);used+=1
        table[bits]=(np.uint32(0x80000000)|np.uint32(node)) if node>=0 else np.uint32(((-node-1)<<5)|used)
    return codes,lengths,table,tree


def pack(weight,hist):
    n=weight.numel();groups=n//128
    codes,lens,table,tree=codes_for(hist)
    maxlen=int(lens.max())
    low=torch.empty((128,groups),device='cuda',dtype=torch.uint8)
    fixed=torch.empty((64,groups),device='cuda',dtype=torch.uint8)
    offsets=np.zeros(groups+1,dtype=np.uint32)
    wires=[];word_count=0;checksum=0;clipped_checksum=0
    base=int(np.convolve(hist,np.ones(16,dtype=np.int64),mode='valid').argmax())
    flat=weight.view(torch.int16).flatten()
    started=time.monotonic()
    for start in range(0,n,262144):
        bits=flat[start:start+262144].cpu().numpy().view(np.uint16).reshape(-1,128)
        ex=((bits>>7)&255).astype(np.int64)
        ls=lens[ex];pos=np.cumsum(ls,axis=1)-ls
        counts=ls.sum(axis=1);words=(counts+31)//32
        g=len(bits)
        bitrows=np.zeros((g,int(words.max())*32),dtype=np.uint8)
        for bit in range(maxlen):
            rows,values=np.nonzero(ls>bit)
            bitrows[rows,pos[rows,values]+bit]=((codes[ex[rows,values]]>>bit)&1).astype(np.uint8)
        byte_rows=np.packbits(bitrows,axis=1,bitorder='little')
        wire=np.concatenate([byte_rows[j,:int(words[j])*4] for j in range(g)])
        first=start//128
        offsets[first:first+g]=word_count+np.concatenate(([0],np.cumsum(words[:-1]))).astype(np.uint32)
        word_count+=int(words.sum());wires.append(wire)
        rawlow=((bits&127)|((bits>>8)&128)).astype(np.uint8)
        low[:,first:first+g]=torch.from_numpy(rawlow.T.copy()).cuda()
        off=np.clip(ex-base,0,15).astype(np.uint8)
        nibble=off[:,0::2]|(off[:,1::2]<<4)
        fixed[:,first:first+g]=torch.from_numpy(nibble.T.copy()).cuda()
        checksum+=int(bits.astype(np.uint64).sum())
        clipped=(rawlow.astype(np.uint16)&127)|((rawlow.astype(np.uint16)&128)<<8)|((base+off.astype(np.uint16))<<7)
        clipped_checksum+=int(clipped.astype(np.uint64).sum())
    offsets[-1]=word_count
    wire=np.concatenate([*wires,np.zeros(4,dtype=np.uint8)])
    gpuwire=torch.from_numpy(wire).cuda()
    gpuparts=[torch.from_numpy(a).cuda() for a in (offsets.view(np.int32),table.view(np.int32),tree)]
    storage=n+wire.nbytes+offsets.nbytes+table.nbytes+tree.nbytes+32
    return dict(low=low,stream=gpuwire,offsets=gpuparts[0],table=gpuparts[1],tree=gpuparts[2],fixed=fixed,
                groups=groups,base=base,checksum=checksum,clipped_checksum=clipped_checksum,
                metadata=dict(total_bytes=storage,bits_per_value=8*storage/n,exponent_bits=float((hist*lens).sum()/hist.sum()),
                              longest_code=maxlen,stream_bytes=wire.nbytes,offset_bytes=offsets.nbytes,
                              pack_seconds=time.monotonic()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='huffman-bound.json')
    parser.add_argument('--seed',type=int,default=5151)
    args=parser.parse_args();outpath=Path(args.output)
    report=dict(status='running',seed=args.seed,rows=[],errors=[],
                scope='Exact Huffman decoder diagnostic, not GEMV. Fixed12 control clips rare exponent tails and omits correction.',
                timing='same captured external events;256MiB cold flush; no subtraction')
    start=time.monotonic()
    try:
        before=time.monotonic()
        ext=load_inline('dense_huffman_'+hashlib.sha256((CPP+CUDA).encode()).hexdigest()[:16],CPP,cuda_sources=CUDA,
                        functions=['decode'],no_implicit_headers=True,with_pytorch_error_handling=False,verbose=True,
                        extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90','-Xptxas=-v'])
        report['compile_seconds']=time.monotonic()-before
        weights=reference.generate_weights(args.seed)
        flush=torch.ones(256*1024*1024//4,device='cuda',dtype=torch.float32)
        blocks=1024;sums=torch.empty(blocks*256,device='cuda',dtype=torch.int32)
        for layer in ('qkv_gate','o','pooler','unpooler','unpooler_prefix1024'):
            w=weights['unpooler'][:1024] if layer=='unpooler_prefix1024' else weights[layer]
            hist=np.zeros(256,dtype=np.int64)
            flat=w.view(torch.int16).flatten()
            for offset in range(0,flat.numel(),262144):
                bits=flat[offset:offset+262144].cpu().numpy().view(np.uint16)
                hist+=np.bincount((bits>>7)&255,minlength=256)
            p=pack(w,hist)
            row=dict(layer=layer,shape=list(w.shape),format=p['metadata'],status='checking',samples=[])
            report['rows'].append(row);screen.save(outpath,report)
            decoded=torch.empty(w.numel(),device='cuda',dtype=torch.int16)
            def call(mode,store=False):
                stream=p['stream'] if mode else p['fixed']
                ext.decode(p['low'].data_ptr(),stream.data_ptr(),p['offsets'].data_ptr(),p['table'].data_ptr(),p['tree'].data_ptr(),sums.data_ptr(),decoded.data_ptr(),p['groups'],p['base'],store,mode,blocks,torch.cuda.current_stream().cuda_stream)
                return sums
            call(1,True);torch.cuda.synchronize()
            row['bit_errors']=int((decoded!=w.view(torch.int16).flatten()).sum())
            if row['bit_errors']:
                raise RuntimeError('Huffman bit reconstruction failed')
            graphs={}
            for mode in (0,1):
                graphs[mode]=screen.capture(lambda d,s,mode=mode:call(mode),{},None,flush)
            for rep in range(36):
                record={}
                for mode in ((0,1) if rep%2==0 else (1,0)):
                    _,graph,_,a,b,_=graphs[mode]
                    graph.replay();b.synchronize()
                    value=int(sums.long().sum())
                    expected=p['checksum'] if mode else p['clipped_checksum']
                    if value!=expected:
                        raise RuntimeError(f'checksum{mode}: {value}!={expected}')
                    record[str(mode)]=a.elapsed_time(b)*1000
                row['samples'].append(record)
            row['fixed12_us']=statistics.median(r['0'] for r in row['samples'])
            row['huffman_us']=statistics.median(r['1'] for r in row['samples'])
            row['median_fixed_over_huffman']=statistics.median(r['0']/r['1'] for r in row['samples'])
            row['status']='passed';screen.save(outpath,report)
            print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
            graphs.clear();del p,decoded
        report['status']='complete'
    except BaseException:
        report['status']='failed';report['errors'].append(traceback.format_exc());raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start;screen.save(outpath,report)
        print(json.dumps(dict(status=report['status'],output=str(outpath))),flush=True)


if __name__=='__main__':
    main()
