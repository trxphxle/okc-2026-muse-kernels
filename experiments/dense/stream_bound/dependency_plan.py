"""Untimed fixture preparation for the dependency-aware stream CEILING ONLY.

Never import a replacement baseline or use this as submission preparation. Real
producer outputs are captured once, with all hooks restored before returning.
Only heavy arithmetic is idealized; dependency glue and routing remain timed in
the companion kernel. Native payload coverage is exactly the validated flat
plan's coverage. CPU payload transfers stay bounded by stream_plan._HASH_BYTES;
expert descriptors reference all 32 original allocations without decoding them.

Launch: ext.dep_launch(plan['params'].data_ptr(), mode, stream.cuda_stream).
The extension launches one 256-thread CTA per SM. Build and call dep_info()
before graph capture on the launch thread/device; keep the stream explicit.
Modes 0/1 and 2/3 share these descriptors; mode4 uses the original flat table.
Keep the entire plan alive through every launch. Reusing it freezes producer
values, not execution dependencies, and requires unchanged source weights/KV.
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from dependency_abi import F, FIELDS, FLAGS, PHASES, arrival_producers
import stream_plan as native_plan
import stream_flat


@triton.jit
def _actual_scores(LOGITS, BIAS, SCORES, IDS):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    logit = tl.load(LOGITS + t * 256 + e)
    bias = tl.load(BIAS + e)
    # Exactly the frozen baseline _sl_tile_route expression and packed tie rule.
    score = tl.sigmoid(logit * 0.1767766952966369) + bias
    tl.store(SCORES + t * 256 + e, score)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
    picked = tl.topk(packed, 8)
    tl.store(IDS + t * 8 + tl.arange(0, 8), (255 - (picked & 255)).to(tl.int32))


def _require(condition, message):
    native_plan._require(condition, message)


def _tensor(tensor, shape, dtype, device, name):
    _require(isinstance(tensor, torch.Tensor), name + ': missing tensor')
    _require(tuple(tensor.shape) == tuple(shape), name + ': unexpected shape ' + str(tuple(tensor.shape)))
    _require(tensor.dtype == dtype and tensor.device == device and tensor.is_contiguous(),
             name + ': wrong dtype/device or noncontiguous storage')
    return tensor


def capture_oracles(base, data, state):
    """Run the baseline once; restore both original functions even on failure.

    References are retained by the hooks and cloned only after the entire call.
    This avoids inserting clone kernels between the baseline's PDL producers.
    This operation is intentionally untimed and not CUDA-graph-capturable.
    """
    tokens = int(data['config']['tokens'])
    _require(tokens in (1, 8), 'only the frozen D1/V1 native small-T paths are supported')
    dense, pool_router = base.dense, base._pool_router_fp32
    held = {}

    def retain(name, value):
        _require(name not in held, 'unexpected repeated oracle producer: ' + name)
        held[name] = value

    def capture_dense(*args, **kwargs):
        result = dense(*args, **kwargs)
        name = args[-1] if args and isinstance(args[-1], str) else kwargs.get('name')
        if name in ('o', 'un'):
            payload, splits = result
            _require(splits == 1 and payload.dtype == torch.bfloat16,
                     name + ': expected native S=1 BF16 producer, refusing fallback')
            retain(name, payload.view(tokens, 8192 if name == 'o' else 1024))
        return result

    def capture_pool(h, hr, T, state_arg, ppick):
        _require(T == tokens and hr is not None, 'missing native FP32 router activation')
        result = pool_router(h, hr, T, state_arg, ppick)
        payload, splits = result
        _require(splits == -1, 'expected native separate (pooled, logits), S=-1')
        pooled, logits = payload
        for name, value in (('h', h), ('hr', hr), ('pooled', pooled), ('logits', logits)):
            retain(name, value)
        return result

    stream0 = data['stream'].detach().clone()
    try:
        base.dense = capture_dense
        base._pool_router_fp32 = capture_pool
        output = base.kernel_fn(data, state)
    finally:
        base.dense = dense
        base._pool_router_fp32 = pool_router
    _require(set(held) == {'o', 'un', 'h', 'hr', 'pooled', 'logits'},
             'baseline did not traverse all required real native producers: ' + str(sorted(held)))
    result = {name: tensor.detach().clone() for name, tensor in held.items()}
    _require(isinstance(output, (tuple, list)) and len(output) == 3, 'unexpected baseline output schema')
    result['output'] = tuple(tensor.detach().clone() for tensor in output)
    result['stream0'] = stream0
    loc = data['cache_loc'].reshape(-1).to(dtype=torch.int64)
    result['new_kv'] = torch.stack(tuple(
        data[name].view(torch.uint8).reshape(-1, 128).index_select(0, loc)
        for name in ('k_cache', 'v_cache')))
    return result


def _fma(a, b, c):
    # Bounded activation-only FP64 temporary emulates one FP32 fused operation.
    # This never touches or converts a dense/expert weight allocation.
    return (a.double() * b.double() + c.double()).float()


def reference_glue(oracles, graft, gprexp):
    """Independent torch equations with both BF16 norm boundaries intact.

    Reductions intentionally use torch rather than reproducing the kernel's
    reduction topology. Actual captured H/HR and final outputs remain primary.
    """
    o = oracles['o'].float()
    n = (o * torch.rsqrt((o * o).sum(-1, keepdim=True) / 8192.0 + 1e-8)).bfloat16().float()
    stream = oracles['stream0']
    alpha, beta = graft['alpha1'], graft['beta1']
    residual = _fma(beta, n, alpha * stream)
    if o.shape[0] > 1:
        # Frozen cluster B1 contracts alpha*stream for the first 4096 columns
        # on the PRELOADED_BASE path, and beta*n for the remaining columns.
        residual[:, :4096] = _fma(alpha[:4096], stream[:, :4096], beta[:4096] * n[:, :4096])
    hr = (residual * torch.rsqrt((residual * residual).sum(-1, keepdim=True) / 8192.0 + 1e-5)) * graft['pre1'].float()
    carry = graft['alpha2'] * residual
    tail = (carry[:, 1024:] * carry[:, 1024:]).sum(-1)
    f = (oracles['un'].float() * graft['post2'][:1024].float()).bfloat16().float()
    norm = (f * torch.rsqrt((f * f).sum(-1, keepdim=True) / 8192.0 + 1e-8)).bfloat16().float()
    head = _fma(graft['beta2'][:1024], norm, carry[:, :1024])
    sout = torch.cat((head, carry[:, 1024:]), dim=-1)
    invnext = torch.rsqrt((tail[:, None] + (head * head).sum(-1, keepdim=True)) / 8192.0 + 1e-5)
    out = ((sout * invnext) * graft['pre2'].float()).bfloat16()
    pooled = oracles['pooled'].float()
    xm = ((pooled * torch.rsqrt((pooled * pooled).sum(-1, keepdim=True) / 4096.0 + 1e-5)) * gprexp.float()).bfloat16()
    return {'H_LIVE': hr.bfloat16(), 'HR_LIVE': hr, 'CARRY': carry, 'TAIL': tail,
            'OUT': out, 'SOUT': sout, 'XM_LIVE': xm, 'RESIDUAL': residual}


def _merged_coverage(chunks):
    intervals = sorted((int(ptr), int(ptr) + int(size)) for ptr, size, _ in chunks)
    merged = []
    for begin, end in intervals:
        _require(begin < end, 'empty payload range')
        if merged and begin == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            _require(not merged or begin > merged[-1][1], 'duplicate/overlapping physical payload')
            merged.append((begin, end))
    return merged


def _raw_xor(tensor):
    raw = tensor.detach().reshape(-1).view(torch.uint8)
    result = 0
    for at in range(0, raw.numel(), native_plan._HASH_BYTES):
        result ^= native_plan._xor(raw[at:at + native_plan._HASH_BYTES].cpu())
    return result


def _validate_routing(scores, gpu_ids, baseline_ids):
    cpu = scores.detach().cpu().numpy()
    _require(bool(np.isfinite(cpu).all()), 'nonfinite actual router scores')
    bits = cpu.view(np.uint32)
    ordered = np.where((bits & np.uint32(0x80000000)) != 0, ~bits,
                       bits ^ np.uint32(0x80000000)).astype(np.uint64)
    keys = (ordered << np.uint64(32)) | (255 - np.arange(256, dtype=np.uint64))
    picked = np.argsort(keys, axis=-1)[:, ::-1][:, :8].astype(np.int32)
    got = gpu_ids.detach().cpu().numpy()
    want = baseline_ids.detach().cpu().numpy()
    _require(np.array_equal(picked, got), 'CPU packed tie ordering disagrees with GPU top8 on actual scores')
    _require(np.array_equal(got, want),
             'actual-logit sigmoid+bias top8 disagrees with captured baseline IDs; refusing fabricated scores: '
             + str({'actual_scores_ids': got.tolist(), 'baseline_ids': want.tolist()}))
    return picked


def make_plan(base, weights, data, state, baseline_output, flat_plan=None):
    """Return one retained plan for strict/prefetch/validation/feature-off modes."""
    tokens = int(data['config']['tokens'])
    _require(tokens in (1, 8), 'D1/V1 only')
    _require(int(data['config']['top_k']) == 8, 'top_k must be eight')
    device = weights['qkv_gate'].device
    _require(device.type == 'cuda', 'CUDA allocations required')
    blocks = torch.cuda.get_device_properties(device).multi_processor_count
    _require(blocks >= tokens * 8, 'one-CTA-per-B2-tile requires at least T*8 resident CTAs')
    oracles = capture_oracles(base, data, state)
    for name, width, dtype in (('o', 8192, torch.bfloat16), ('un', 1024, torch.bfloat16),
                               ('h', 8192, torch.bfloat16), ('hr', 8192, torch.float32),
                               ('pooled', 4096, torch.bfloat16), ('logits', 256, torch.float32),
                               ('stream0', 8192, torch.float32)):
        _tensor(oracles[name], (tokens, width), dtype, device, name)
    _, baseline_routes = native_plan._routes(baseline_output, data)
    touched, routing = native_plan._routes(oracles['output'], data)
    _require(routing['global_ids'] == baseline_routes['global_ids'], 'oracle capture changed baseline routing')
    if flat_plan is None:
        flat_plan = stream_flat.make_plan(weights, data, oracles['output'], state, representation='native')
    _require(flat_plan['manifest']['representation'] == 'native', 'flat plan must use native source bytes')
    _require(flat_plan['manifest']['tokens'] == tokens, 'flat plan token count mismatch')
    _require(flat_plan['manifest']['routing']['global_ids'] == routing['global_ids'], 'flat plan routing mismatch')
    flat_cpu = np.asarray(flat_plan['chunks_cpu'], dtype=np.int64)
    _require(bool((flat_cpu[:, 2] == 0).all()), 'FLAT_CHUNKS must be the original flattened zero-offset table')
    entries = flat_plan['manifest']['sources']
    by_name = {entry['name']: entry for entry in entries}
    _require(len(by_name) == len(entries), 'duplicate flat source names')
    phase_for = {
        'o': 2, 'pooler': 4, 'router': 4, 'unpooler': 7,
        'attn_residual_gate': 3, 'pre_ffn_norm': 3, 'ffn_residual_gate': 3,
        'router_bias': 5, 'pre_expert_norm': 5, 'post_expert_norm': 6,
        'post_ffn_norm': 8, 'next_pre_attn_norm': 8, 'w13_gs': 6, 'w2_gs': 6,
        'k_cache': 1, 'v_cache': 1, 'kv_scale_static_packet': 0,
    }
    expert_names = ('w13', 'w13_sf', 'w2', 'w2_sf')
    _require(set(by_name) == set(phase_for) | set(expert_names) | {'qkv_gate'}, 'unaccounted native source')
    qkv = weights['qkv_gate']
    _tensor(qkv, (2304, 8192), torch.bfloat16, device, 'qkv_gate')
    qkv_split = qkv.data_ptr() + 1280 * 8192 * 2
    per_phase = [[] for _ in PHASES]
    source_totals = {name: 0 for name in by_name}
    # Source allocations are disjoint in the validated native plan; each source
    # chunk belongs to exactly one entry, including scattered physical KV runs.
    spans = sorted((entry['source_address'], entry['source_address'] + entry['allocation_view_bytes'],
                    entry['name']) for entry in entries)
    starts = np.asarray([span[0] for span in spans], dtype=np.int64)
    for ptr, size, _ in flat_cpu:
        ptr, size = int(ptr), int(size)
        index = int(np.searchsorted(starts, ptr, side='right')) - 1
        _require(index >= 0, 'flat chunk outside known sources')
        begin, end, name = spans[index]
        _require(begin <= ptr and ptr + size <= end, 'flat chunk crosses native source allocation')
        source_totals[name] += size
        if name in expert_names:
            continue
        if name == 'qkv_gate':
            if ptr < qkv_split:
                first = min(size, qkv_split - ptr)
                per_phase[0].append((ptr, first, 0))
                ptr, size = ptr + first, size - first
            if size:
                per_phase[1].append((ptr, size, 0))
        else:
            per_phase[phase_for[name]].append((ptr, size, 0))
    for name, total in source_totals.items():
        _require(total == by_name[name]['physical_bytes'], name + ': flat source byte total mismatch')
    for name, width in (('o', 2048), ('pooler', 16384), ('router', 32768), ('unpooler', 8192)):
        entry = by_name[name]
        for ptr, size, _ in per_phase[phase_for[name]]:
            if entry['source_address'] <= ptr < entry['source_address'] + entry['allocation_view_bytes']:
                _require((ptr - entry['source_address']) % width == 0 and size % width == 0,
                         name + ': dense producer descriptors must contain complete full-K rows')
    expert_bank = []
    for expert in range(32):
        chunks = []
        for name in expert_names:
            tensor = weights[name]
            _require(tensor.device == device and tensor.is_contiguous() and tensor.shape[0] == 32
                     and tensor.element_size() == 1, name + ': expected 32 contiguous native packed experts')
            size = tensor[expert].numel()
            _require(size % 32768 == 0, name + ': per-expert allocation must have complete 32KiB stages')
            ptr = tensor.data_ptr() + expert * size
            chunks.extend((ptr + offset, 32768, 0) for offset in range(0, size, 32768))
        _require(len(chunks) == 864, 'expert bank must hold exactly 28,311,552 bytes / 864 chunks per expert')
        expert_bank.append(chunks)
    bank_cpu = np.asarray(expert_bank, dtype=np.int64)
    selected = [chunk for expert in touched for chunk in expert_bank[int(expert)]]
    fixed = [chunk for phase in per_phase for chunk in phase]
    _require(_merged_coverage(fixed + selected) == _merged_coverage(flat_cpu),
             'dependency fixed+GPU-selected bank coverage differs from flat native coverage')
    expected_bytes = [sum(chunk[1] for chunk in phase) for phase in per_phase]
    expected_bytes[6] += len(touched) * 28311552
    expected_xor = [0] * len(PHASES)
    qkv_xor = (_raw_xor(qkv[:1280]), _raw_xor(qkv[1280:]))
    _require(qkv_xor[0] ^ qkv_xor[1] == by_name['qkv_gate']['expected_xor_u32'],
             'flat QKV bytes changed since validation')
    expected_xor[0], expected_xor[1] = qkv_xor
    for name, entry in by_name.items():
        if name != 'qkv_gate':
            phase = 6 if name in expert_names else phase_for[name]
            expected_xor[phase] ^= entry['expected_xor_u32']
    total_xor = 0
    for value in expected_xor:
        total_xor ^= value
    _require(sum(expected_bytes) == flat_plan['expected_bytes'] and total_xor == flat_plan['expected_xor_u32'],
             'dependency bytes/XOR disagree with validated flat oracle')
    ranges, offset = [], 0
    for phase in per_phase:
        ranges.append((offset, len(phase)))
        offset += len(phase)
    chunks_cpu = np.asarray(fixed, dtype=np.int64).reshape(-1, 3)
    ranges_cpu = np.asarray(ranges, dtype=np.int64)
    graft = state['graft']
    for name in ('alpha1', 'beta1', 'pre1', 'alpha2', 'beta2', 'post2', 'pre2'):
        dtype = torch.float32 if name.startswith(('alpha', 'beta')) else torch.bfloat16
        _tensor(graft[name], (8192,), dtype, device, 'graft.' + name)
    _tensor(state['bias'], (256,), torch.float32, device, 'baseline bias')
    _require(torch.equal(state['bias'], weights['router_bias']), 'prepared bias differs from native fixture bias')
    scores = torch.empty((tokens, 256), dtype=torch.float32, device=device)
    expected_ids = torch.empty((tokens, 8), dtype=torch.int32, device=device)
    _actual_scores[(tokens,)](oracles['logits'], state['bias'], scores, expected_ids,
                              num_warps=base.SL_TILE_CONFIG[tokens][0])
    ids_cpu = _validate_routing(scores, expected_ids, oracles['output'][2])
    expected_mask = sum(1 << int(expert) for expert in touched)
    gprexp = (1.0 + weights['pre_expert_norm'].float()).bfloat16().contiguous()
    _tensor(gprexp, (4096,), torch.bfloat16, device, 'effective pre-expert norm')
    torch_reference = reference_glue(oracles, graft, gprexp)
    tensors = {
        'CHUNKS': torch.from_numpy(chunks_cpu).to(device),
        'RANGES': torch.from_numpy(ranges_cpu).to(device),
        'EXP_CHUNKS': torch.from_numpy(bank_cpu).to(device),
        'LOCAL_START': data['local_start'].reshape(1).to(dtype=torch.int32).contiguous(),
        'K_CACHE': data['k_cache'], 'V_CACHE': data['v_cache'],
        'CACHE_LOC': data['cache_loc'].reshape(tokens).to(dtype=torch.int64).contiguous(),
        'NEW_KV': oracles['new_kv'], 'O_ORACLE': oracles['o'], 'POOLED_ORACLE': oracles['pooled'],
        'SCORES_ORACLE': scores, 'UN_ORACLE': oracles['un'], 'STREAM0': oracles['stream0'],
        'GPREXP': gprexp, 'O_W': weights['o'], 'POOL_W': weights['pooler'],
        'ROUTER_W': weights['router'], 'UN_W': weights['unpooler'],
        'FLAT_CHUNKS': flat_plan['chunks'],
    }
    tensors.update({name.upper(): graft[name] for name in ('alpha1', 'beta1', 'pre1', 'alpha2', 'beta2', 'post2', 'pre2')})
    for name, shape, dtype in (
        ('O_LIVE', (tokens, 8192), torch.bfloat16), ('POOLED_LIVE', (tokens, 4096), torch.bfloat16),
        ('SCORES_LIVE', (tokens, 256), torch.float32), ('UN_LIVE', (tokens, 1024), torch.bfloat16),
        ('H_LIVE', (tokens, 8192), torch.bfloat16), ('HR_LIVE', (tokens, 8192), torch.float32),
        ('CARRY', (tokens, 8192), torch.float32), ('TAIL', (tokens,), torch.float32),
        ('OUT', (tokens, 8192), torch.bfloat16), ('SOUT', (tokens, 8192), torch.float32),
        ('IDS_OUT', (tokens, 8), torch.int32), ('XM_LIVE', (tokens, 4096), torch.bfloat16),
        ('SINK', (blocks, 9, 2), torch.int64), ('STAMPS', (blocks, 9, 3), torch.int64),
        ('READY_TIMES', (len(FLAGS),), torch.int64)):
        tensors[name] = torch.empty(shape, dtype=dtype, device=device)
    for name, shape, dtype in (('FLAGS', (len(FLAGS),), torch.uint32), ('ARRIVALS', (len(FLAGS), blocks, 32), torch.uint32),
                               ('ACTIVE_MASK', (1,), torch.uint32), ('ERROR', (1,), torch.int32)):
        tensors[name] = torch.zeros(shape, dtype=dtype, device=device)
    scalars = {'EXP_CHUNKS_PER': 864, 'T': tokens, 'BLOCKS': blocks,
               'DELAY_STAGE': -1, 'FLAT_COUNT': len(flat_cpu)}
    _require(set(tensors) | set(scalars) == set(FIELDS) and not set(tensors) & set(scalars),
             'planner does not populate the complete shared ABI')
    values = [0] * len(FIELDS)
    for name, tensor in tensors.items():
        values[F[name]] = tensor.data_ptr()
    for name, value in scalars.items():
        values[F[name]] = value
    params = torch.tensor(values, dtype=torch.int64, device=device)
    references = {name: value for name, value in torch_reference.items() if name in tensors}
    references.update({'O_LIVE': oracles['o'], 'POOLED_LIVE': oracles['pooled'], 'SCORES_LIVE': scores,
                       'UN_LIVE': oracles['un'], 'H_LIVE': oracles['h'], 'HR_LIVE': oracles['hr'],
                       'OUT': oracles['output'][0], 'SOUT': oracles['output'][1],
                       'IDS_OUT': expected_ids, 'NEW_KV': oracles['new_kv']})
    exact = ('O_LIVE', 'POOLED_LIVE', 'SCORES_LIVE', 'UN_LIVE', 'IDS_OUT', 'NEW_KV')
    tolerances = {name: {'kind': 'exact', 'rtol': 0.0, 'atol': 0.0} for name in exact}
    for name in ('H_LIVE', 'OUT', 'XM_LIVE'):
        tolerances[name] = {'kind': 'bf16_adjacent', 'max_bf16_steps': 1,
                            'absolute_floor_fp32_eps': 8, 'max_nonexact_fraction': 0.01,
                            'note': 'adjacent representable BF16 value, or absolute error <= 8*float32 epsilon'}
    for name in ('HR_LIVE', 'CARRY', 'SOUT'):
        tolerances[name] = {'kind': 'allclose', 'rtol': 2e-5, 'atol': 2e-6}
    tolerances['TAIL'] = {'kind': 'allclose', 'rtol': 2e-5, 'atol': 2e-6}
    manifest = {
        'format': 'dependency_native_ceiling', 'oracle_only': True, 'tokens': tokens, 'blocks': blocks,
        'fields': list(FIELDS), 'phases': list(PHASES), 'flags': list(FLAGS), 'scalars': scalars,
        'kernel_launch': {'entry': 'dep_launch',
                          'arguments': ["params.data_ptr()", 'mode', 'stream.cuda_stream'],
                          'threads_per_cta': 256, 'ctas': blocks},
        'tensor_schema': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                          for name, value in tensors.items()},
        'expected_bytes': sum(expected_bytes), 'expected_payload_bytes': sum(expected_bytes),
        'expected_xor_u32': total_xor, 'expected_bytes_per_phase': expected_bytes,
        'expected_xor_per_phase': expected_xor, 'phase_fixed_chunk_counts': [len(p) for p in per_phase],
        'routing': routing, 'expected_mask': expected_mask,
        'actual_score_top8': {'cpu_gpu_baseline_agree': True, 'ids': ids_cpu.tolist(),
                              'expression': 'tl.sigmoid(logit * 0.1767766952966369) + bias',
                              'tie_order': 'descending ordered float bits, ascending global expert ID'},
        'expert_bank': {'experts': 32, 'chunks_per_expert': 864, 'bytes_per_expert': 28311552,
                        'descriptor_storage_bytes': int(bank_cpu.nbytes),
                        'selection': 'timed GPU top8 active-local bitmap only; no host selected bank'},
        'coverage': {'flat_exact_physical_ranges': True, 'flat_exact_payload_bytes': True,
                     'flat_exact_xor': True, 'duplicate_physical_bytes': 0,
                     'qkv_rows_phase0': [0, 1280], 'gate_rows_phase1': [1280, 2304],
                     'full_k_dense_rows': True, 'full_causal_kv_union': True},
        'flat_manifest': flat_plan['manifest'],
        'metadata': {'fixed_descriptor_bytes': int(chunks_cpu.nbytes + ranges_cpu.nbytes),
                     'expert_bank_descriptor_bytes': int(bank_cpu.nbytes), 'params_bytes': len(FIELDS) * 8,
                     'local_start_cache_loc_bytes': 4 + 8 * tokens,
                     'readiness_state_bytes': (len(FLAGS) + len(FLAGS) * blocks * 32) * 4 + 8,
                     'validation_sink_timestamp_bytes': blocks * 9 * (2 + 3) * 8 + len(FLAGS) * 8,
                     'new_kv_producer_bytes': 2 * tokens * 128,
                     'graft_parameter_storage_bytes': sum(tensors[name].numel() * tensors[name].element_size()
                         for name in ('ALPHA1', 'BETA1', 'PRE1', 'ALPHA2', 'BETA2', 'POST2', 'PRE2', 'GPREXP')),
                     'extra_traffic': 'descriptor, readiness, producer-copy, glue activation and parameter traffic is additional to flat payload; measure physical DRAM with NCU',
                     'cpu_payload_batch_bytes': native_plan._HASH_BYTES},
        'reference_sources': {'H_LIVE': 'actual baseline pool/router input', 'HR_LIVE': 'actual baseline unrounded FP32 router input',
                              'OUT': 'actual baseline output', 'SOUT': 'actual baseline output',
                              'CARRY': 'independent torch B1', 'TAIL': 'independent torch B1', 'XM_LIVE': 'independent torch pre-expert norm'},
        'validation_tolerances': tolerances,
        'validation_contract': {'arrival_producers': arrival_producers(tokens, blocks), 'error': 0,
                                'flags_epoch': 'all flags equal one incremented invocation epoch',
                                'active_mask': expected_mask, 'byte_xor_sink_reduce_axes': [0],
                                'kv_first_issue_not_before': 'READY_TIMES[KV_WRITTEN]',
                                'baseline_and_capture_ids_exact': True},
        'idealizations': ['frozen REAL O/pooler/logit/unpooler and newKV producer payloads',
                          'dense GEMM, attention and expert arithmetic omitted',
                          'exact static physical KV coverage oracle; no timed page traversal',
                          'native raw gate/norm bytes streamed plus effective prepared glue parameters',
                          'full global-scale vectors only when present in validated flat conservative footprint'],
    }
    keepalive = [weights, data, state, flat_plan, oracles, torch_reference, references, params, tensors, expected_ids]
    return {'params': params, 'tensors': tensors, 'scalars': scalars, 'keepalive': keepalive,
            'manifest': manifest, 'expected_ids': expected_ids, 'expected_mask': expected_mask,
            'expected_bytes_per_phase': expected_bytes, 'expected_xor_per_phase': expected_xor,
            'expected_payload_bytes': sum(expected_bytes), 'expected_xor_u32': total_xor,
            'flat_plan': flat_plan, 'references': references, 'torch_reference': torch_reference,
            'oracles': oracles, 'chunks_cpu': chunks_cpu, 'ranges_cpu': ranges_cpu,
            'expert_chunks_cpu': bank_cpu}
