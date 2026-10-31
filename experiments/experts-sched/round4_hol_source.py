"""Diagnostic-only source surgery for round4_hol; never edits the frozen decoder.

Only lane-zero timestamp stores, metadata stores and a host resource query are
added. In particular EMPTY barriers still receive all sixteen warp arrivals.
Runtime p.dbg=nullptr is the feature-off arm of the *same* compiled module.
"""
import hashlib

FIELDS = 16
ACTORS = 18
HEADER = 8
FOOTER = 9  # eight original DBG locations, then per-actor overflow flag


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'Diagnostic anchor expected once: {old[:100]!r}')
    return source.replace(old, new, 1)


def instrument(source, capacity):
    helpers = r'''
// External Round4 diagnostic. No atomics or new shared memory/barriers.
constexpr int R4_CAP = @CAP@;
constexpr int R4_STRIDE = 9 + R4_CAP * 16;
__device__ __forceinline__ unsigned long long* r4_actor(const Params& p) {
  if ((threadIdx.x & 31) || !p.dbg) return nullptr;
  const int first = (int)p.dbg[0], count = (int)p.dbg[1];
  const int c = (int)blockIdx.x - first;
  if (c < 0 || c >= count) return nullptr;
  return p.dbg + 8 + ((size_t)c * 18 + (threadIdx.x >> 5)) * R4_STRIDE;
}
__device__ __forceinline__ void r4_put(unsigned long long* a, int seq, int field,
                                      unsigned long long value, bool clock = true) {
  if (!a) return;
  if (seq < 0 || seq >= R4_CAP) { a[8] = 1; return; }
  // Compiler memory clobbers prevent movement across the observed operation.
  unsigned long long t = value;
  if (clock) asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  reinterpret_cast<volatile unsigned long long*>(a)[9 + (size_t)seq * 16 + field] = t;
}
__device__ __forceinline__ void r4_mark(const Params& p, int field) {
  auto a = r4_actor(p);
  if (!a) return;
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  reinterpret_cast<volatile unsigned long long*>(a)[field] = t;
}
#define DBG(k) r4_mark(p, k)
'''.replace('@CAP@', str(capacity))
    old = '#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)'
    source = replace_once(source, old, helpers)
    # Producers map the exact dynamically reordered stage stream, not a host guess.
    for ring, nslots, slot, phase, bias in [('W', 'WGS_NS', 'ws', 'wph', 1000),
                                           ('X', 'WGS_NX', 'xs', 'xph', 2000000)]:
        old = f'      if (n >= {nslots}) wgs_wait(S.bars + B{ring}_E + {slot}, {phase} ^ 1, {bias} + gs);'
        new = ('      r4_put(p, n, 14, ((unsigned long long)pb << 32) | (unsigned)(gs + 1), false);\n'
               f'      if (n >= {nslots}) {{\n'
               '        r4_put(p, n, 0, 0);\n'
               f'        wgs_wait(S.bars + B{ring}_E + {slot}, {phase} ^ 1, {bias} + gs);\n'
               '        r4_put(p, n, 1, 0);\n      }')
        source = replace_once(source, old, new)
    old = '        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }'
    source = replace_once(source, old,
        '        if (!pb && !(OKC_WGSX & 8)) { r4_put(p, n, 2, 0); while (ld_acquire(p.xready + j) < npad) __nanosleep(32); r4_put(p, n, 3, 0); }')
    old = '      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }'
    source = replace_once(source, old,
        '      if (pb && !(OKC_WGSX & 8)) { r4_put(p, n, 4, 0); while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); r4_put(p, n, 5, 0); }')
    for ring, slot, limit, ph in [('W', 'ws', 'WGS_NS', 'wph'), ('X', 'xs', 'WGS_NX', 'xph')]:
        anchor = f'      uint64_t* bar = S.bars + B{ring}_F + {slot};'
        source = replace_once(source, anchor, '      r4_put(p, n, 6, 0);\n' + anchor)
        anchor = f'      if (++{slot} == {limit}) {{ {slot} = 0; {ph} ^= 1; }}'
        source = replace_once(source, anchor, '      r4_put(p, n, 7, 0);\n' + anchor)
    for ring, field, slot, limit, bias in [('W', 0, 'ws', 'WGS_NS', 3000000),
                                          ('X', 2, 'xs', 'WGS_NX', 4000000)]:
        old = f'    wgs_wait(S.bars + B{ring}_F + {slot}, (uint32_t)(seq / {limit}) & 1u, {bias} + seq);'
        source = replace_once(source, old, f'    r4_put(p, seq, {field}, 0);\n' + old +
                              f'\n    r4_put(p, seq, {field + 1}, 0);')
    old = '        mbar_arrive1(S.bars + BW_E + ws);'
    source = replace_once(source, old,
        '        r4_put(p, seq, 4, 0);\n' + old + '\n        r4_put(p, seq, 5, 0);')
    # Both X release sites release seq-1: seq has advanced at segment exit,
    # and the deferred previous-stage release is inside the next stage.
    old = 'if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);'
    if source.count(old) != 2:
        raise RuntimeError('Expected exactly two X release sites')
    source = source.replace(old,
        'if (lane == 0) { r4_put(p, seq - 1, 6, 0); mbar_arrive1(S.bars + BX_E + prev); r4_put(p, seq - 1, 7, 0); }')
    def flush_block(indent):
        return ('{\n' + indent + '  r4_put(p, seq, 10, ((unsigned long long)PB << 32) | ((unsigned long long)(pd.u + 1) << 4) | pd.kind, false);\n' +
                indent + '  r4_put(p, seq, 8, 0);\n' +
                indent + '  wgs_flush<PB>(p, S, pd);\n' +
                indent + '  r4_put(p, seq, 9, 0);\n' + indent + '}')
    old = '        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);'
    source = replace_once(source, old, '        if (FIRST && ci == OKC_WGSFCI && pd.kind) ' + flush_block('        '))
    source = replace_once(source, '  wgs_flush<PB>(p, S, pd);\n}',
                          '  if (pd.kind) ' + flush_block('  ') + '\n}')
    # Unit end includes the WGMMA drain, X release, partial red.add or epilogue.
    start = source.index('__device__ __forceinline__ void wgs_seg(')
    end = source.index('template <bool PB>\n__device__ __forceinline__ void wgs_phase', start)
    segment = source[start:end]
    segment = replace_once(segment, '  wg_wait<0>();',
        '  r4_put(p, seq, 13, ((unsigned long long)PB << 32) | (unsigned)(u + 1), false);\n'
        '  r4_put(p, seq, 11, 0);\n  wg_wait<0>();')
    segment = segment.replace('return;', 'r4_put(p, seq, 12, 0); return;')
    anchor = '  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred\n}'
    segment = replace_once(segment, anchor, anchor[:-1] + '  r4_put(p, seq, 12, 0);\n}')
    source = source[:start] + segment + source[end:]
    # Cohort headers are immutable during a replay. Resolve the actor pointer
    # once per producer stream/consumer segment, not twice at every stamp.
    # This removes avoidable global header loads from the measured hot loop.
    for opener, closer, anchor in (
            ('__device__ void wgs_wproducer(', '// x producer:',
             '  if ((threadIdx.x & 31) != 0) return;'),
            ('__device__ void wgs_xproducer(', "// phase A's x gather",
             '  if ((threadIdx.x & 31) != 0) return;'),
            ('__device__ __forceinline__ void wgs_seg(',
             'template <bool PB>\n__device__ __forceinline__ void wgs_phase',
             '  const int tid = threadIdx.x, lane = tid & 31;')):
        begin = source.index(opener)
        end = source.index(closer, begin)
        block = replace_once(source[begin:end], anchor,
                             anchor + '\n  auto* r4a = r4_actor(p);')
        block = block.replace('r4_put(p,', 'r4_put(r4a,')
        source = source[:begin] + block + source[end:]
    # Only rare phase-tail stamps remain outside those hot scopes.
    source = source.replace('r4_put(p,', 'r4_put(r4_actor(p),')
    return source


RESOURCE_CPP = '\nstd::vector<int64_t> r4_resources();\n'
RESOURCE_CUDA = r'''
std::vector<int64_t> r4_resources() {
  cudaFuncAttributes a{};
  const size_t sm = okc_smem_bytes();
  TORCH_CHECK(cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) == cudaSuccess,
              "resource query dynamic shared-memory attribute failed");
  TORCH_CHECK(cudaFuncGetAttributes(&a, okc::moe_fused<true>) == cudaSuccess, "resource attribute query failed");
  int blocks = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, okc::moe_fused<true>, okc::NTL, sm) == cudaSuccess,
              "resource occupancy query failed");
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes,
          (int64_t)sm, blocks, okc::NTL, okc::NW, okc::WGS_NS, okc::WGS_NX};
}
'''
RESOURCE_KEYS = ('registers_per_thread', 'local_bytes_per_thread', 'static_shared_bytes',
                 'dynamic_shared_bytes', 'active_blocks_per_sm', 'block_threads',
                 'consumer_warps', 'weight_slots', 'activation_slots')


def build(module, state, capacity):
    from torch.utils.cpp_extension import load_inline
    import torch
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (9, 0):
        raise RuntimeError('This fixed 132-CTA diagnostic requires the H200 sm90 lane')
    defines = tuple(module._main_defines(tuple(state['ma_sfx']))) + tuple(state['ma_sfx'])
    if module.DQ7 and '-DOKC_DQ7=1' not in defines:
        defines += ('-DOKC_DQ7=1',)
    defaults = []
    for key, value in [('NS', module._MA_MAIN_NS), ('FNST', 4), ('NSF', 3)]:
        if not any(d.startswith(f'-DOKC_{key}=') for d in defines):
            defaults.append(f'-DOKC_{key}={value}')
    flags = ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', *defaults, *defines]
    modules, metadata = {}, {}
    for name, cuda in [('original', module.MOE_CUDA_SRC),
                       ('instrumented', instrument(module.MOE_CUDA_SRC, capacity))]:
        cpp, cuda = module.MOE_CPP_SRC + RESOURCE_CPP, cuda + RESOURCE_CUDA
        digest = hashlib.sha256((cpp + cuda + repr(flags)).encode()).hexdigest()
        ext = load_inline(name='r4_hol_' + digest[:24], cpp_sources=cpp,
            cuda_sources=cuda, functions=['okc_run', 'okc_set_pdl', 'r4_resources'],
            extra_cuda_cflags=flags, verbose=False)
        # No -std override: use the installed Torch C++20 default.
        ext.okc_set_pdl(0)
        modules[name] = ext
        metadata[name] = dict(source_sha256=digest, cuda_flags=flags,
            resources=dict(zip(RESOURCE_KEYS, map(int, ext.r4_resources()))))
    return modules, metadata
