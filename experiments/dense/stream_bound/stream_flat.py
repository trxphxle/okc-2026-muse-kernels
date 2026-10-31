"""Ceiling-reader variant: original exact chunks become independent ring stages.
No weight/KV data is copied or repacked. The unused stage-header tensor exists
only for shared host validation; this kernel reads the direct chunk table only.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
import torch
from stream_kernel import CPP, CPP_FLAGS, CUDA_FLAGS
import stream_plan as native_plan


def flatten(plan):
    old_chunks,old_stages=plan['chunks'],plan['stages']
    chunks=plan['chunks_cpu'].copy()
    chunks[:,2]=0
    stages=np.column_stack((np.arange(len(chunks),dtype=np.int64),np.ones(len(chunks),dtype=np.int64),chunks[:,1]))
    gpu_chunks=torch.from_numpy(chunks).to(old_chunks.device)
    gpu_stages=torch.from_numpy(stages).to(old_chunks.device)
    plan['chunks']=gpu_chunks;plan['stages']=gpu_stages
    plan['chunks_cpu']=chunks;plan['stages_cpu']=stages
    if 'keepalive' in plan:
        plan['keepalive']=[x for x in plan['keepalive'] if x is not old_chunks and x is not old_stages]
        plan['keepalive'].extend((gpu_chunks,gpu_stages))
    if 'manifest' in plan:
        manifest=plan['manifest']
        manifest['reader_layout']='one original physical chunk per ring stage, striped across all CTAs'
        manifest['grouped_stage_count']=manifest['stage_count']
        manifest['stage_count']=len(chunks)
        manifest['metadata']['descriptor_storage_bytes']=chunks.nbytes
        manifest['metadata']['unused_validation_stage_storage_bytes']=stages.nbytes
        manifest['metadata']['descriptor_traffic']='Only direct chunk records are read; stages tensor is host-validation bookkeeping, no GPU reads.'
        manifest['coverage']['identical_payload_ranges_to_grouped_reader']=True
    return plan


def make_plan(weights,data,output,state=None,representation='native',stage_bytes=32768):
    return flatten(native_plan.make_plan(weights,data,output,state,representation,stage_bytes))


@lru_cache(maxsize=1)
def build():
    from torch.utils.cpp_extension import load_inline
    source=Path(__file__).with_suffix('.cu').read_text()
    digest=hashlib.sha256(('\0'.join((CPP,source,*CPP_FLAGS,*CUDA_FLAGS))).encode()).hexdigest()[:20]
    ext=load_inline('dense_flat_stream_'+digest,CPP,cuda_sources=source,functions=['stream_launch','stream_info'],
                    extra_cflags=list(CPP_FLAGS),extra_cuda_cflags=list(CUDA_FLAGS),
                    no_implicit_headers=True,with_pytorch_error_handling=False,verbose=True)
    ext.stream_info(0);ext.stream_info(1)
    return ext
