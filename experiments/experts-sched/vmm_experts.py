"""Diagnostic-only WGS backing comparison; no expert math or layout changes.

The two existing WGS banks contain their original NVFP4 codes and scales.
Recommended allocation granularity is an alignment contract, not evidence of
physical PTE size. Public MoE reference computes in FP32 and returns BF16.
Only the native isolated expert stage is timed; no full-layer claim is made.
"""

import torch


_ROUTE_SEEDS = (741283, 741284, 741285)
_BANKS = (12, 13)


def _fixture(module, state, arena_w, snapshots, moe_reference, ref_weights,
             m, seed, flush, time_arms, exact):
    data = moe_reference.generate_input(bs=m, sharding='ep8', rank='busiest', seed=seed)
    cfg = data['config']
    if cfg['local_start'] != 0 or cfg['local_count'] != 32:
        raise RuntimeError('Expected public busiest local32 routing')
    x, ids, tw = (data[k] for k in ('hidden_states', 'topk_ids', 'topk_weights'))
    inputs = (x, ids, tw)
    input_snapshots = tuple(t.clone() for t in inputs)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        gold = moe_reference.ref_kernel(data, ref_weights)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    if not bool(torch.isfinite(gold).all()):
        raise RuntimeError('Nonfinite public MoE reference')
    counts = torch.bincount(ids.long().flatten(), minlength=256)[:32].cpu().tolist()
    native = module._moe(m, state)
    native.okc_set_pdl(0)
    current_w = state['ma_w']
    scratch, nsm = state['ma_scratch'], state['nsm']
    arms = {}
    for name, w in [('current', current_w), ('current_repeat', current_w)] + (
            [('arena', arena_w)] if arena_w is not None else []):
        out = torch.empty_like(x)

        def run(w=w, out=out):
            native.okc_run(w, scratch, x, ids, tw, out, 0, 32, nsm)
            return out

        arms[name] = run

    def validate(outputs):
        input_equal = {name: bool(exact(t, saved)) for name, t, saved in zip(
            ('hidden_states', 'topk_ids', 'topk_weights'), inputs, input_snapshots)}
        weight_equal = {str(i): bool(exact(current_w[i], saved))
                        for i, saved in snapshots.items()}
        arena_equal = ({str(i): bool(exact(arena_w[i], snapshots[i])) for i in _BANKS}
                       if arena_w is not None else {})
        numerical = {}
        for name, output in outputs.items():
            ok, detail = moe_reference.compare(output, gold, moe_reference.REL_L2_TOL)
            diff = output.float() - gold.float()
            numerical[name] = dict(
                passed=bool(ok), detail=str(detail),
                relative_l2=float(diff.norm() / gold.float().norm().clamp(min=1e-12)),
                max_abs=float(diff.abs().max()))
        pairwise = {}
        for name in outputs:
            if name == 'current':
                continue
            other = outputs[name]
            diff = other.float() - outputs['current'].float()
            pairwise[name] = dict(
                bitwise_equal=bool(exact(outputs['current'], other)),
                relative_l2=float(diff.norm() / outputs['current'].float().norm().clamp(min=1e-12)),
                max_abs=float(diff.abs().max()))
        return dict(input_bytes_equal=input_equal, current_weight_bytes_equal=weight_equal,
                    arena_weight_bytes_equal=arena_equal, reference=numerical,
                    versus_current=pairwise,
                    passed=all(input_equal.values()) and all(weight_equal.values())
                    and all(arena_equal.values()) and all(v['passed'] for v in numerical.values()))

    result = time_arms(arms, flush, base='current', repeat='current_repeat',
                       validate=validate, reps=36)
    result.update(m=m, load='V16' if m == 128 else 'V32', routing_seed=seed,
                  local_counts=counts, local_pairs=sum(counts),
                  distinct_local_experts=sum(c > 0 for c in counts),
                  comparison_available=arena_w is not None)
    reliable = (result['aa_within_one_percent'] and result['checks_before']['passed']
                and result['checks_after']['passed'])
    result['reliable'] = bool(reliable)
    result['stage_gain_gate_met'] = bool(
        arena_w is not None and reliable and result['median_paired_ratio']['arena'] >= 1.03)
    return result


def run_experts(module, weights, state, ext, moe_reference, flush, time_arms,
                exact, report, checkpoint):
    """Run six fixed routing fixtures, retaining only JSON evidence on return."""
    entry = dict(status='running', cases=[], route_seeds=list(_ROUTE_SEEDS),
                 allocation_compression_requested=False, stage_gain_gate=1.03,
                 reference='public MoE FP32 arithmetic, BF16 output, original decode local32 weights',
                 reference_tolerance=float(moe_reference.REL_L2_TOL),
                 granularity_caveat='CUDA recommended allocation granularity is not proven physical PTE size',
                 bitwise_caveat='Output bit identity is reported separately from public tolerance; reductions may be nondeterministic',
                 recommendation='none; diagnostic-only')
    report['experts'] = entry
    current_w = state['ma_w']
    if len(current_w) <= 13:
        raise RuntimeError('Active WGS banks ma_w[12:14] are required')
    entry['native_defines'] = list(module._main_defines(state['ma_sfx']))
    if '-DOKC_WGS=1' not in entry['native_defines']:
        raise RuntimeError('The requested V16/V32 WGS path is not selected for these scales')
    for i in _BANKS:
        if not current_w[i].is_contiguous():
            raise RuntimeError('WGS banks must be contiguous')
    ref_weights = {k: weights[k] for k in ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
    ref_weights['post_norm_w'] = weights['post_expert_norm']
    if any(ref_weights[k].shape[0] != 32 for k in ('w13', 'w13_sf', 'w2', 'w2_sf')):
        raise RuntimeError('Expected original decode local32 expert weights')
    # One bounded snapshot per active bank, plus the small ordinary global scales
    # and norm. These witnesses catch source mutations, not just two copies that
    # happen to change identically. Inactive alternate layouts are never passed
    # differently, and scratch deliberately retains native reset/reuse behavior.
    snapshots = {i: current_w[i].clone() for i in (2, 5, 7, 12, 13)}
    arena, arena_w = None, None
    caps = ext.capabilities()
    recommended = int(caps.get('normal_recommended_granularity', 0))
    sizes = [current_w[i].numel() * current_w[i].element_size() for i in _BANKS]
    entry['banks'] = [dict(index=i, bytes=size, shape=list(current_w[i].shape),
                           dtype=str(current_w[i].dtype), current_ptr=int(current_w[i].data_ptr()),
                           current_ptr_mod_recommended=(int(current_w[i].data_ptr()) % recommended
                                                        if recommended else None))
                      for i, size in zip(_BANKS, sizes)]
    try:
        if not caps.get('vmm_supported', False):
            entry['allocation_status'] = 'vmm_unsupported'
        elif recommended <= 0:
            raise RuntimeError('Normal recommended allocation granularity is required')
        else:
            offsets = [0, ((sizes[0] + recommended - 1) // recommended) * recommended]
            nbytes = offsets[1] + sizes[1]
            entry.update(arena_offsets=offsets, arena_logical_bytes=nbytes,
                         normal_min_granularity=caps.get('normal_min_granularity'),
                         normal_recommended_granularity=recommended)
            try:
                arena, metadata = ext.allocate(nbytes, False, True)
            except RuntimeError as exc:
                entry.update(allocation_status='normal_vmm_allocation_denied',
                             allocation_error=str(exc))
            else:
                entry['allocation_metadata'] = metadata
                entry['arena_ptr_mod_recommended'] = int(arena.data_ptr()) % recommended
                if entry['arena_ptr_mod_recommended'] != 0:
                    raise RuntimeError('Arena VA is not recommended-granularity aligned')
                arena_w = list(current_w)
                for i, size, offset, bank in zip(_BANKS, sizes, offsets, entry['banks']):
                    view = arena.narrow(0, offset, size).view(current_w[i].dtype).view(current_w[i].shape)
                    ext.copy_into(view, current_w[i])
                    arena_w[i] = view
                    bank.update(arena_offset=offset, arena_ptr_mod_recommended=int(view.data_ptr()) % recommended)
                del view
                if not all(arena_w[i] is current_w[i] for i in range(len(current_w)) if i not in _BANKS):
                    raise RuntimeError('Non-WGS weight identity changed')
                entry['allocation_status'] = 'normal_vmm_arena'
        for m in (128, 256):
            for seed in _ROUTE_SEEDS:
                result = _fixture(module, state, arena_w, snapshots, moe_reference, ref_weights,
                                  m, seed, flush, time_arms, exact)
                entry['cases'].append(result)
                checkpoint()
                if not result['checks_before']['passed'] or not result['checks_after']['passed']:
                    raise RuntimeError('Expert byte identity or public reference validation failed')
        entry['stage_gain_gate_met_all_six'] = all(c['stage_gain_gate_met'] for c in entry['cases'])
        entry['recommendation'] = ('diagnostic stage gate met on all six fixtures; no official/full-layer claim'
                                   if entry['stage_gain_gate_met_all_six'] else 'no reliable >=3% all-fixture stage gain')
        entry['status'] = 'complete' if arena_w is not None else 'complete_without_vmm_comparison'
        checkpoint()
    finally:
        # Graphs and callables are fixture-local and gone before arena release.
        # Neither the report nor prepared state ever owns an arena-backed tensor.
        torch.cuda.synchronize()
        if arena_w is not None:
            arena_w.clear()
        arena = None
        snapshots.clear()
