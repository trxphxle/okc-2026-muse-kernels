"""Diagnostic only: native attention overlap, A/A, serialization, deletion ceiling.

NEVER submit this script. The delete_attention arm deliberately reuses a fixed
attention result to bound attention's entire exposed cost, not legal speedup.
The copied decoder remains byte-identical to the specified public baseline.
Timing/trace helpers are the existing compare_decode_public.py implementation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
import reference
import compare_decode_public as bench

bench.torch = torch
bench.reference = reference


def overlaps(a, b):
    return max(0.0, min(a['end_us'], b['end_us']) - max(a['start_us'], b['start_us']))


def attention_summary(trace):
    launches = trace['launches']
    attention = [x for x in launches if 'attn_kernel<' in x['name']]
    if len(attention) != 1:
        raise RuntimeError(f'Expected one native attention kernel, got {len(attention)}')
    attn = attention[0]
    others = [x for x in launches if x is not attn]
    simultaneous = [dict(name=x['name'], overlap_us=overlaps(attn, x),
                         start_us=x['start_us'], duration_us=x['duration_us'])
                    for x in others if overlaps(attn, x) > 0]
    before = [x for x in others if x['start_us'] < attn['start_us']]
    after = [x for x in others if x['start_us'] >= attn['start_us']]
    previous = max(before, key=lambda x: x['start_us']) if before else None
    following = min(after, key=lambda x: x['start_us']) if after else None
    return dict(attention=attn, overlapping_kernels=simultaneous,
                attention_overlap_union_us=bench.union([
                    (max(attn['start_us'], x['start_us']), min(attn['end_us'], x['end_us']))
                    for x in others if overlaps(attn, x) > 0]),
                previous=previous, following=following,
                previous_gap_us=attn['start_us'] - previous['end_us'] if previous else None,
                following_gap_us=following['start_us'] - attn['end_us'] if following else None)


def profile(fn, data, state, flush, path):
    captured = bench.capture(fn, data, state)
    flush.sum()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        captured['graph'].replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    trace = bench.summarize_trace(path)
    return dict(trace=trace, attention=attention_summary(trace))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='decode_control.py')
    parser.add_argument('--output', default='overlap.json')
    parser.add_argument('--seed', type=int, default=731283)
    parser.add_argument('--input-offset', type=int, default=0)
    parser.add_argument('--pairs', type=int, default=15)
    parser.add_argument('--profiles', type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    prefix = output.stem
    report = dict(kind='attention_overlap_ceiling_diagnostic', submittable=False,
                  source_sha256=bench.sha(args.baseline), seed=args.seed,
                  input_offset=args.input_offset, pairs=args.pairs, profiles=args.profiles,
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name()),
                  timing_protocol='256 MiB cold-L2 sum inside graph, followed by captured external timing events; unprofiled paired samples',
                  precision_caveat='Specified attribution-only baseline is not precision-qualified; oracle tolerance is not precision legality.',
                  artifact_files=[], results=[], status='running')

    def save():
        bench.save(output, report)

    started = time.monotonic()
    save()
    try:
        module = bench.load(args.baseline, 'overlap_control')
        weights = reference.generate_weights(args.seed)
        prepare_start = time.monotonic()
        state = module.prepare(weights)
        report['prepare_seconds'] = time.monotonic() - prepare_start
        report['pdl_source_policy'] = dict(PDL=module.PDL, PDL_MAX_T=module.PDL_MAX_T,
                                         V16=module._pdl(128), V32=module._pdl(256))
        report['state_modes'] = {k: bench.metadata(state.get(k)) for k in
                                 ('att_ok', 'mm_pick', 'moe_pick', 'graft_lt', 'compose_graft_enabled')}
        flush = torch.ones(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
        original_decode = module._decode_attention7
        original_launch = module._launch7
        save()
        for load, batch in (('V16', 16), ('V32', 32)):
            for layer in ('sliding', 'full'):
                seed = 919233 + batch * 100 + 8000 + (layer == 'full') * 10000 + args.input_offset
                shape = dict(batch=batch, q_len=8, ctx=80000, layer=layer, page=1,
                             spread='agentic', dist='soft')
                data = reference.generate_input(seed=seed, **shape)
                eager = module.kernel_fn(data, state)
                counts = torch.bincount((eager[2].long() // 32).flatten(), minlength=8)
                data['local_start'].fill_(32 * int(counts.argmax()))
                pristine = bench.clone_data(data)
                extracted = {}

                def extract(d, st, qkv):
                    result = original_decode(d, st, qkv)
                    extracted['qkv'] = qkv.detach().clone()
                    extracted['attention'] = result.detach().clone()
                    return result

                module._decode_attention7 = extract
                try:
                    eager = module.kernel_fn(data, state)
                    torch.cuda.synchronize()
                finally:
                    module._decode_attention7 = original_decode
                if not extracted:
                    raise RuntimeError('Baseline did not execute the specified native attention path')
                row = dict(load=load, layer=layer, input_seed=seed, shape=shape,
                           local_start=int(data['local_start']), records=[], native={})
                report['results'].append(row)
                row['oracle'] = bench.oracle(data, eager, weights, pristine)
                row['attention_plan'] = module._plan7(batch, 1, 8, data['config']['window'],
                                                     data['page_table'].shape[1], 1, data['attn_input'].device)
                graphs = {}
                graphs['baseline'] = bench.capture(module.kernel_fn, data, state, flush)
                graphs['baseline_repeat'] = bench.capture(module.kernel_fn, data, state, flush)

                def serial_launch(d, **kwargs):
                    kwargs['pdl'] = False
                    return original_launch(d, **kwargs)

                module._launch7 = serial_launch
                try:
                    graphs['serial_attention'] = bench.capture(module.kernel_fn, data, state, flush)
                finally:
                    module._launch7 = original_launch

                def deleted_attention(d, st, qkv):
                    return extracted['attention']

                module._decode_attention7 = deleted_attention
                try:
                    graphs['delete_attention'] = bench.capture(module.kernel_fn, data, state, flush)
                finally:
                    module._decode_attention7 = original_decode

                def isolated(d, st):
                    return original_decode(d, st, extracted['qkv'])

                graphs['isolated_attention'] = bench.capture(isolated, data, state, flush)
                names = tuple(graphs)
                for pair in range(args.pairs):
                    rotated = names[pair % len(names):] + names[:pair % len(names)]
                    order = rotated if pair % 2 == 0 else rotated[::-1]
                    times = {name: bench.measure(graphs[name]) for name in order}
                    row['records'].append(dict(pair=pair, order=order, us=times))
                row['median_us'] = {name: statistics.median(r['us'][name] for r in row['records']) for name in names}
                row['comparisons'] = {name: bench.comparison(graphs[name]['output'], graphs['baseline']['output'])
                                      for name in ('baseline_repeat', 'serial_attention', 'delete_attention')}
                row['arm_oracles'] = {name: bench.oracle(data, graphs[name]['output'], weights, pristine)
                                      for name in ('baseline', 'baseline_repeat', 'serial_attention', 'delete_attention')}
                row['arm_routes'] = {name: bench.routes(graphs[name]['output'], row['local_start'])
                                     for name in ('baseline', 'baseline_repeat', 'serial_attention', 'delete_attention')}
                isolated_out = graphs['isolated_attention']['output']
                expected_attn = extracted['attention'].float()
                delta = isolated_out.float() - expected_attn
                norms = expected_attn.norm(dim=-1)
                row['isolated_comparison'] = dict(
                    bitwise_equal=bool(torch.equal(isolated_out, extracted['attention'])),
                    finite=bool(torch.isfinite(isolated_out).all()),
                    l2=float(delta.norm() / expected_attn.norm().clamp_min(1e-12)),
                    row_l2=float((delta.norm(dim=-1) / norms.clamp_min(float(norms.mean()) * 0.1).clamp_min(1e-12)).max()))
                if not all(c['passed'] for c in row['arm_oracles'].values()):
                    raise RuntimeError('Public oracle rejected a fixed-input diagnostic arm')
                isolated_check = row['isolated_comparison']
                if not isolated_check['finite'] or isolated_check['l2'] > 0.003 or isolated_check['row_l2'] > 0.015:
                    raise RuntimeError('Isolated attention exceeded baseline-agreement tolerance')
                del pristine, isolated_out, expected_attn, delta, norms
                times = row['median_us']
                row['aa_ratio'] = times['baseline'] / times['baseline_repeat']
                row['serial_ratio'] = times['baseline'] / times['serial_attention']
                row['attention_deletion_us'] = times['baseline'] - times['delete_attention']
                row['attention_deletion_fraction'] = row['attention_deletion_us'] / times['baseline']
                save()
                for arm in ('baseline', 'serial_attention', 'isolated_attention'):
                    module._launch7 = serial_launch if arm == 'serial_attention' else original_launch
                    fn = isolated if arm == 'isolated_attention' else module.kernel_fn
                    try:
                        row['native'][arm] = []
                        for repeat in range(args.profiles):
                            filename = f'{prefix}-{load}-{layer}-{arm}-{repeat}-native.json'
                            measured = profile(fn, data, state, flush, Path(filename))
                            row['native'][arm].append(measured)
                            report['artifact_files'].append(filename)
                            save()
                    finally:
                        module._launch7 = original_launch
                duration = {arm: statistics.median(x['attention']['attention']['duration_us'] for x in traces)
                            for arm, traces in row['native'].items()}
                overlap = max(x['attention']['attention_overlap_union_us']
                              for x in row['native']['baseline'])
                row['native_attention_median_us'] = duration
                row['max_baseline_attention_overlap_us'] = overlap
                row['in_layer_minus_isolated_attention_us'] = duration['baseline'] - duration['isolated_attention']
                # A co-runner can gain at most its observed overlapped interval;
                # absence of any concurrent kernel excludes this scheduling lever.
                row['observed_corunner_overlap_ceiling_us'] = overlap
                print(json.dumps({k: row[k] for k in ('load', 'layer', 'median_us', 'aa_ratio', 'serial_ratio',
                      'attention_deletion_us', 'native_attention_median_us', 'max_baseline_attention_overlap_us')}), flush=True)
                save()
                del graphs, eager, data, extracted
                gc.collect()
        report['aa_accepted'] = all(abs(r['aa_ratio'] - 1) <= 0.01 for r in report['results'])
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        save()
        print(json.dumps(dict(status=report['status'], output=str(output),
                              elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
