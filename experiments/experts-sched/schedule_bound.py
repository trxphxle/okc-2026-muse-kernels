"""External WGS scheduling-slack diagnostic, never a submission or optimization.

Run in /workspace with screen_moe.py and the public reference.py/checks.py/
nvfp4.py alongside the unchanged baseline:
  python schedule_bound.py --baseline muse_moe_w4a16.py --output schedule-bound.json

One prepare reuses the baseline's four modules and packed state. Only two extra
MAIN modules are built: existing DBG stamps and DBG plus per-consumer-warp
completion observers. CUDA remains the attributed alygg77/vjeux Apache-2.0
source loaded from the baseline; no consumer arithmetic or scheduling changes.
The only source edits enable the otherwise hard-coded-off OKC_DEBUG macro and
insert two syncwarp/leader timestamp stores in spare debug rows. Their maxima
cover all 16 consumer warps, not just thread 0. They measure instruction-side
completion, not a new GPU-scope fence or visibility guarantee. Producers are
not included in completion maxima. Existing kernel barriers remain unchanged.

Max-minus-mean finish slack is an optimistic *model*, not a rigorous scheduling
upper bound or a measured speedup: phase A/B overlap, dependencies, throughput,
unit granularity and observer code generation can defeat additive savings.
Zero savings on D1/D16/V1 restricts the model to a WGS-only scheduling change.
All six loads enter the denominator equally. Native-span and event-minus-4us
sensitivities are explicitly hypothetical, not judge-adjusted measurements.
The unchanged baseline may have independent precision-policy issues; passing
the public numeric oracle here does not establish submission compliance.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import statistics
import threading
import time
import traceback

import torch
import screen_moe as screen


WGS_LOADS = {'D32', 'V16', 'V32'}
STAMPS = ['entry', 'routed', 'A_thread0', 'B_thread0',
          'post_grid_barrier', 'final_thread0', 'post_gather_join', 'post_gather']


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Baseline source anchor changed: ' + old)
    return source.replace(old, new, 1)


def debug_modules(module, state):
    defines = tuple(module._main_defines(state['sfx'])) + state['sfx']
    if '-DOKC_WGS=1' not in defines or '-DOKC_NW=16' not in defines:
        raise RuntimeError('Required 16-warp WGS MAIN is not selected')
    source = module.CUDA_SRC
    for define in defines:
        if define.startswith('@'):
            source = module.ALT_SRCS[define[1:]]
    source = replace_once(source, '#define OKC_DEBUG 0',
                          '#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif')
    observed = source
    for phase, call in enumerate((
            'wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);',
            'wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);')):
        observed = replace_once(observed, call, call + '\n'
            '    // Diagnostic only: all consumer warps, no cross-warp barrier.\n'
            '    __syncwarp();\n'
            '    if (p.dbg && (threadIdx.x & 31) == 0)\n'
            '      p.dbg[gridDim.x * 8 + ' + str(phase) +
            ' * gridDim.x * 16 + blockIdx.x * 16 + threadIdx.x / 32] = gtimer();')
    modules, identities = {}, {}
    for name, text in [('debug', source), ('observer', observed)]:
        digest = hashlib.sha256(text.encode()).hexdigest()
        tag = 'schedule_bound_' + name + '_' + digest[:16]
        module.ALT_SRCS[tag] = text
        flags = tuple(d for d in defines if not d.startswith('@')) + (
            '-DOKC_DEBUG=1', '@' + tag)
        modules[name] = module._ext(flags)
        identities[name] = {'source_sha256': digest, 'defines': list(flags)}
    return modules, identities


def debug_function(module, ext):
    def run(data, state):
        module._bind_context(data['hidden_states'])
        cfg = data['config']
        if cfg['local_start'] != state['lo'] or cfg['local_count'] != state['cnt']:
            raise ValueError('Public input shard differs from prepared shard')
        x = data['hidden_states'].contiguous()
        ids = data['topk_ids'].to(torch.int32).contiguous()
        if ids.data_ptr() & 15:
            ids = ids.clone()
        tw = data['topk_weights'].to(torch.float32).contiguous()
        out = torch.empty_like(x, dtype=torch.bfloat16)
        if x.shape[0] > module.CHUNK_TOKENS:
            raise ValueError('Diagnostic requires one native launch')
        ext.okc_run(state['w'], state['scratch'], x, ids, tw, out,
                    state['lo'], state['cnt'], state['grid'])
        return out
    return run


def stamp_metrics(raw, consumers, event_us):
    errors = []
    if any(v <= 0 for row in raw for v in row):
        errors.append('unset or nonpositive Gx8 stamp')
    for row in raw:
        if any(row[b] < row[a] for a, b in
               ((0, 1), (1, 7), (7, 6), (6, 2), (2, 3), (3, 4), (4, 5))):
            errors.append('nonmonotonic native stamps')
            break
    origin = min(row[0] for row in raw)
    native = (max(row[5] for row in raw) - origin) / 1000
    if not 0 < native <= event_us + 2.0:
        errors.append('native span outside event envelope (2us tolerance)')
    finishes = [[row[2] for row in raw], [row[3] for row in raw]]
    if consumers is not None:
        if any(v <= 0 for phase in consumers for row in phase for v in row):
            errors.append('unset consumer stamp')
        finishes = [[max(row) for row in phase] for phase in consumers]
        for c, row in enumerate(raw):
            if not (row[6] <= finishes[0][c] <= finishes[1][c] <= row[4]):
                errors.append('consumer finish outside phase/barrier envelope')
                break
    # Subtract integer timestamps before averaging: globaltimer may be near
    # epoch-nanosecond magnitude, where float cancellation loses ~256 ns.
    slack = [statistics.mean(max(values) - v for v in values) / 1000 for values in finishes]
    return dict(valid=not errors, errors=errors, native_span_us=native,
                native_span_is_lower_bound=True, phase_A_slack_us=slack[0],
                phase_B_slack_us=slack[1], both_slack_us=sum(slack),
                phase_finish_us=[[ (v-origin)/1000 for v in phase] for phase in finishes],
                thread0_underestimate_us=[max((finishes[p][c]-raw[c][2+p])/1000
                                             for c in range(len(raw))) for p in range(2)])


def summarize(report):
    rows = report['loads']
    if len(rows) != len(screen.LOADS) or any(not r.get('complete') for r in rows):
        return {'conclusion': 'incomplete; no scheduling conclusion'}
    reliable = True
    summaries = []
    for row in rows:
        records = row['records']
        aa = statistics.median(r['us']['base'] / r['us']['base_repeat'] for r in records)
        base = statistics.median(r['us']['base'] for r in records)
        item = dict(load=row['load'], baseline_us=base, repeat_ratio=aa)
        reliable &= abs(aa - 1) <= .01
        if row['load'] in WGS_LOADS:
            overhead = {name: statistics.median(r['us'][name] / r['us']['base'] - 1
                                                for r in records)
                        for name in ('debug', 'observer')}
            valid = all(r['stamps'][name]['metrics']['valid'] for r in records
                        for name in ('debug', 'observer'))
            reliable &= valid and all(abs(v) <= .03 for v in overhead.values())
            item.update(stamp_overhead_fraction=overhead, stamps_valid=valid)
        summaries.append(item)
    models = {}
    for phase, key in [('B_only', 'phase_B_slack_us'), ('A_plus_B', 'both_slack_us')]:
        for denominator in ('raw_event', 'event_minus4us', 'native_span_sensitivity'):
            ratios, optimistic, discounted = [], [], []
            for row in rows:
                samples, high_samples, low_samples = [], [], []
                for r in row['records']:
                    if row['load'] not in WGS_LOADS:
                        samples.append(1.0); high_samples.append(1.0); low_samples.append(1.0)
                        continue
                    metrics = r['stamps']['observer']['metrics']
                    base = r['us']['base']
                    denom = (base if denominator == 'raw_event' else base - 4
                             if denominator == 'event_minus4us' else metrics['native_span_us'])
                    slack = metrics[key]
                    noise = abs(base - r['us']['base_repeat'])
                    perturbation = abs(r['us']['observer'] - base)
                    def ratio(saving):
                        if denom <= 0 or saving >= denom:
                            return None
                        return denom / (denom - saving)
                    samples.append(ratio(slack))
                    high_samples.append(ratio(slack + noise + perturbation))
                    low_samples.append(ratio(max(0, slack - noise - perturbation)))
                if any(x is None for x in samples + high_samples + low_samples):
                    reliable = False
                    ratios = []
                    break
                ratios.append(statistics.median(samples))
                optimistic.append(max(high_samples))
                discounted.append(min(low_samples))
            models[phase + '/' + denominator] = (
                dict(hypothetical_speedup=screen.geomean(ratios),
                     hypothetical_latency_reduction=1 - 1 / screen.geomean(ratios),
                     noise_and_observer_inflated_speedup=screen.geomean(optimistic),
                     noise_and_observer_discounted_speedup=screen.geomean(discounted),
                     exceeds_3pct_speedup_gate=screen.geomean(discounted) >= 1.03)
                if ratios else {'invalid_denominator': True})
    main = models['A_plus_B/raw_event']
    if not reliable:
        conclusion = 'no strong conclusion: observer perturbation, A/A, or stamp sanity failed'
    elif main['noise_and_observer_discounted_speedup'] >= 1.03:
        conclusion = 'optimistic model clears 3% gate; candidate still required to establish any real gain'
    elif main['noise_and_observer_inflated_speedup'] < 1.03:
        conclusion = 'measured finish-slack model does not clear 3% gate (not a universal scheduling bound)'
    else:
        conclusion = '3% gate unresolved after noise/observer allowance'
    return dict(conclusion=conclusion, reliable_observation=reliable, loads=summaries,
                hypothetical_models=models, gate_speedup=1.03,
                limitations='Allowances are sensitivity bands, not statistical confidence intervals. '
                'A/B slack need not be additive or transferable from observed to unstamped code.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='muse_moe_w4a16.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--weight-seed', type=int, default=731283)
    parser.add_argument('--routing-seeds', default='1234,5678,9012')
    parser.add_argument('--reps', type=int, default=9)
    parser.add_argument('--timeout-seconds', type=int, default=840)
    args = parser.parse_args()
    seeds = [int(s) for s in args.routing_seeds.split(',')]
    if len(set(seeds)) < 3 or args.reps < 9 or not 1 <= args.timeout_seconds <= 870:
        parser.error('Require >=3 distinct routing seeds, >=9 reps/seed, timeout 1..870s')
    if Path(args.output).name != args.output:
        parser.error('--output must be a basename')
    output = Path('/workspace') / args.output
    started = time.monotonic()
    report = dict(status='starting', diagnostic_only=True, arguments=vars(args),
                  stamp_columns=STAMPS, loads=[], errors=[],
                  observer='Spare 32G int64 rows: [phase=2][CTA=G][consumer warp=16]. '
                  'syncwarp plus one globaltimer/store per warp after each phase; no arithmetic edits.',
                  timing='Cold 256MiB read flush outside CUDA events; warmed one-call CUDA graphs. '
                  'Same prepared state, independent baseline-repeat capture and output. '
                  'Public oracle computed once per fresh routing seed; every replay checked.')
    lock = threading.RLock()
    def save():
        with lock:
            report['elapsed_seconds'] = time.monotonic() - started
            screen.save(output, report)
    def expired():
        with lock:
            report['status'] = 'timeout'
            report['errors'].append('Hard wall-clock deadline reached; incomplete evidence')
            save()
        os._exit(124)
    timer = threading.Timer(args.timeout_seconds, expired)
    timer.daemon = True
    timer.start()
    save()
    try:
        reference = screen.load(Path('/workspace/reference.py'), 'schedule_reference')
        path = screen.source_path(args.baseline)
        report['baseline_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        module = screen.load(path, 'schedule_baseline')
        report['device'] = torch.cuda.get_device_name()
        weights = reference.generate_weights(args.weight_seed)
        report['status'] = 'preparing_baseline'; save()
        before = time.monotonic()
        state = module.prepare(weights)
        report['prepare_seconds'] = time.monotonic() - before
        report['status'] = 'compiling_observers'; save()
        before = time.monotonic()
        modules, report['debug_builds'] = debug_modules(module, state)
        report['debug_compile_seconds'] = time.monotonic() - before
        grid = state['grid']
        report['grid'] = grid
        flush = torch.ones(screen.FLUSH_BYTES // 4, dtype=torch.float32, device='cuda')
        buffers = {name: torch.zeros(grid * 40, dtype=torch.int64, device='cuda')
                   for name in modules}
        for load_index, (load, m) in enumerate(screen.LOADS):
            row = dict(load=load, m=m, records=[], routing=[], complete=False)
            report['loads'].append(row)
            report['status'] = 'measuring_' + load; save()
            names = ['base', 'base_repeat'] + (['debug', 'observer'] if load in WGS_LOADS else [])
            captures, events = {}, {}
            # First fixture establishes shapes only; every seed is copied into
            # these stable graph inputs, including subsequent changed routing.
            initial = reference.generate_input(bs=m, sharding='ep8', rank='busiest', seed=seeds[0])
            for name in names:
                data = screen.stable_input(initial)
                if name in modules:
                    debug_state = dict(state, scratch=list(state['scratch']) + [buffers[name]])
                    fn, selected_state = debug_function(module, modules[name]), debug_state
                else:
                    fn, selected_state = module.kernel_fn, state
                graph, result = screen.capture(fn, data, selected_state)
                captures[name] = (data, graph, result)
                events[name] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for event in events[name]:
                    event.record()
            torch.cuda.synchronize()
            for seed_index, seed in enumerate(seeds):
                fresh = reference.generate_input(bs=m, sharding='ep8', rank='busiest', seed=seed)
                route = screen.routing(fresh)
                row['routing'].append(dict(seed=seed, **route))
                gold = reference.ref_kernel(fresh, weights)
                if not bool(torch.isfinite(gold).all()):
                    raise RuntimeError('Nonfinite public reference')
                for data, _, _ in captures.values():
                    screen.copy_input(data, fresh)
                for rep in range(args.reps):
                    shift = (rep + seed_index + load_index) % len(names)
                    order = names[shift:] + names[:shift]
                    if (rep // len(names)) % 2:
                        order = list(reversed(order))
                    # JSON serialization/host inspection can idle the GPU and
                    # penalize the first arm. Warm all arms before the triplet;
                    # each timed replay still starts with the cold-L2 flush.
                    for _ in range(3):
                        for name in order:
                            captures[name][1].replay()
                    torch.cuda.synchronize()
                    record = dict(seed=seed, rep=rep, order=order, us={}, stamps={}, correctness={})
                    row['records'].append(record)
                    for name in order:
                        if name in buffers:
                            buffers[name].zero_()
                        sink = flush.sum()
                        begin, end = events[name]
                        begin.record(); captures[name][1].replay(); end.record()
                        end.synchronize()
                        us = begin.elapsed_time(end) * 1000
                        if not math.isfinite(us) or us <= 0:
                            raise RuntimeError('Invalid CUDA event measurement')
                        record['us'][name] = us
                        del sink
                    # No oracle/copies/host inspection between timed arms.
                    for name in names:
                        result = captures[name][2]
                        passed, message = reference.compare(result, gold, screen.REL_L2_TOL)
                        same, same_message = reference.compare(result, captures['base'][2], screen.REL_L2_TOL)
                        exact = torch.equal(result, captures['base'][2])
                        finite = bool(torch.isfinite(result).all())
                        record['correctness'][name] = dict(public_pass=bool(passed),
                            public_message=str(message), exact_vs_unstamped=exact, finite=finite,
                            unstamped_pass=bool(same), unstamped_message=str(same_message))
                        if name in buffers:
                            host = buffers[name].cpu()
                            raw = host[:grid * 8].view(grid, 8).tolist()
                            consumer = (host[grid * 8:].view(2, grid, 16).tolist()
                                        if name == 'observer' else None)
                            record['stamps'][name] = dict(raw_Gx8_ns=raw,
                                consumer_finish_ns=consumer,
                                metrics=stamp_metrics(raw, consumer, record['us'][name]))
                        if not passed or not same or not finite:
                            save()
                            raise RuntimeError('Fresh replay correctness failed: ' + name)
                    record['status'] = 'passed'; save()
                del gold, fresh
            row['complete'] = True
            del captures, events, data, graph, result, initial
            save()
        report['summary'] = summarize(report)
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'error'
        report['errors'].append(traceback.format_exc())
        raise
    finally:
        save()
        timer.cancel()


if __name__ == '__main__':
    main()
