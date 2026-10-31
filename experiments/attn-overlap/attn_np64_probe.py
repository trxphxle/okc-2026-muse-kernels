"""DIAGNOSTIC ONLY: one-change producer-width prototype, never submit this wrapper.

Uses the approved precision-repaired embedded attention without changing math,
precision, cache addressing, stages, grid, or reduction protocol. The sole CUDA
change is OKC_NP8=64 instead of128 for the flat q_len8 producer team. Attribution
of the derived CUDA remains the vjeux/alygg77 Apache-2.0 notices in decode_current.
"""
import hashlib
import os

import torch
import decode_current as control

_MODULE = None


def module():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline
        old = '#define OKC_NP8 128'
        if control.ATT7_CUDA_SRC.count(old) != 1:
            raise RuntimeError('Producer-width baseline anchor changed')
        source = control.ATT7_CUDA_SRC.replace(old, '#define OKC_NP8 64', 1)
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}' + ('a' if major == 9 else '')
        flags = ['-DOKC_QDIRECT=0', '-DOKC_PK2=0', '-DOKC_W0ENABLE=1', '-O3',
                 f'-DOKC_SM90={int(major >= 9)}', '--expt-relaxed-constexpr']
        name = 'round2_np64_' + hashlib.sha256(source.encode()).hexdigest()[:16]
        _MODULE = load_inline(name=name, cpp_sources=control.ATT7_CPP_SRC,
                              cuda_sources=source, functions=['run'],
                              extra_cuda_cflags=flags, verbose=False)
    return _MODULE


def kernel_fn(data):
    cfg = data['config']
    changed = cfg['q_len'] == 8 and cfg['batch'] > 1 and cfg['window'] == 0
    return control._launch7(data, _mod=module() if changed else None, pdl=False)
