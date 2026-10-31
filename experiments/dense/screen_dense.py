"""External dense-projection screen, adapted from work/bench/screen_moe.py
and d1_cost_ladder.py. Never a submission or an official score.

All five projections are summed within each of the six scored loads before
geometric aggregation. Captured external events exclude graph replay startup;
a 256 MiB read-only flush precedes the events INSIDE each graph. No subtraction.
A/A shares the exact prepared state, with independent input/output graphs.
Different sources prepare independent states; base_repeat always shares base.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

LOADS = (('D1', 1), ('D16', 16), ('D32', 32), ('V1', 8), ('V16', 128), ('V32', 256))
ARMS = ('base', 'candidate', 'base_repeat')
ORDERS = (ARMS, ('candidate', 'base_repeat', 'base'), ('base_repeat', 'base', 'candidate'),
          ('base_repeat', 'candidate', 'base'), ('base', 'base_repeat', 'candidate'),
          ('candidate', 'base', 'base_repeat'))
NEARBY = ((7, 'pooler', 'normal'), (33, 'o', 'normal'), (130, 'unpooler', 'normal'),
          (5, 'unpooler', 'normal'), (8, 'router', 'normal'), (9, 'router', 'outliers'),
          (512, 'qkv_gate', 'outliers'), (1000, 'unpooler', 'outliers'))


def load(path, role):
    spec = importlib.util.spec_from_file_location('_dense_screen_' + role, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save(path, report):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def capture(fn, data, state, flush):
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(data, state)
        begin.record()
        end.record()
    current.wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        sink = flush.sum()
        begin.record()
        out = fn(data, state)
        end.record()
    current.wait_stream(stream)
    for _ in range(5):
        graph.replay()
        end.synchronize()
        if begin.elapsed_time(end) <= 0:
            raise RuntimeError('Invalid captured timing event')
    return data, graph, out, begin, end, sink


def check(reference, fresh, out, gold):
    router = fresh['config']['layer'] == 'router'
    passed, message = reference.compare(out, gold, reference.ROUTER_TOL if router else reference.REL_L2_TOL)
    rows = reference.probe_rows(fresh['x'])
    probe_error = 0.0
    if passed and len(rows):
        err = (out[rows].double() - gold[rows].double()).norm(dim=-1)
        err /= gold[rows].double().norm(dim=-1).clamp_min(1e-30)
        probe_error = float(err.max())
        passed = probe_error <= reference.PROBE_TOL
    return dict(passed=bool(passed), message=message, probe_rows=len(rows), probe_relative_l2=probe_error)


def run_case(reference, modules, states, weights, flush, row, pairs, seed):
    m, layer, dist = row['m'], row['layer'], row['dist']
    initial = reference.generate_input(m=m, layer=layer, seed=seed, dist=dist)
    graphs = {}
    for name in ARMS:
        role = 'candidate' if name == 'candidate' else 'base'
        data = {'x': initial['x'].clone(), 'config': dict(initial['config'])}
        graphs[name] = capture(modules[role].kernel_fn, data, states[role], flush)
    for rep in range(pairs):
        fresh = reference.generate_input(m=m, layer=layer, seed=seed + rep + 1, dist=dist)
        gold = reference.ref_kernel(fresh, weights)
        for capture_data, *_ in graphs.values():
            capture_data['x'].copy_(fresh['x'])
        record = dict(input_seed=seed + rep + 1, us={}, checks={}, order=list(ORDERS[rep % 6]))
        row['records'].append(record)
        for name in record['order']:
            _, graph, _, begin, end, _ = graphs[name]
            graph.replay()
            end.synchronize()
            us = begin.elapsed_time(end) * 1000
            if not math.isfinite(us) or us <= 0:
                raise RuntimeError('Invalid elapsed time: ' + str(us))
            record['us'][name] = us
        for name in ARMS:
            result = check(reference, fresh, graphs[name][2], gold)
            record['checks'][name] = result
            if not result['passed']:
                raise RuntimeError(f'{name} {layer} m={m}: {result}')
        record['ratio'] = {name: record['us']['base'] / record['us'][name]
                           for name in ('candidate', 'base_repeat')}
    row['median_us'] = {name: statistics.median(r['us'][name] for r in row['records']) for name in ARMS}
    row['median_ratio'] = {name: statistics.median(r['ratio'][name] for r in row['records'])
                           for name in ('candidate', 'base_repeat')}
    row['status'] = 'passed'
    graphs.clear()
    gc.collect()


def summarize(screen):
    loads = {}
    for load_name, m in LOADS:
        rows = [r for r in screen['rows'] if r['m'] == m and r['scored']]
        if len(rows) != 5 or any(r['status'] != 'passed' for r in rows):
            raise RuntimeError('Incomplete scored load: ' + load_name)
        base_us = sum(r['median_us']['base'] for r in rows)
        estimated = {'base': base_us}
        for name in ('candidate', 'base_repeat'):
            estimated[name] = sum(r['median_us']['base'] / r['median_ratio'][name] for r in rows)
        loads[load_name] = dict(m=m, summed_us=estimated,
                                ratio={name: base_us / estimated[name] for name in ('candidate', 'base_repeat')})
    screen['loads'] = loads
    screen['geomean_ratio'] = {name: geomean(l['ratio'][name] for l in loads.values())
                               for name in ('candidate', 'base_repeat')}
    screen['control_ok'] = all(abs(l['ratio']['base_repeat'] - 1) <= 0.01 for l in loads.values())
    screen['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='muse_bf16_proj.py')
    parser.add_argument('--candidate')
    parser.add_argument('--output', default='dense-aa.json')
    parser.add_argument('--weight-seeds', default='5151')
    parser.add_argument('--pads-mib', default='0')
    parser.add_argument('--pairs', type=int, default=12)
    parser.add_argument('--nearby', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    report = dict(kind='dense_paired_internal_events', official_score=False, status='running', screens=[],
                  cache_protocol='256MiB read-only sum in graph before external begin event',
                  timing_protocol='captured external CUDA events around kernel_fn; no subtraction',
                  aggregation='per-shape median paired ratios, sum five equivalent latencies per load, geomean six loads',
                  pairs=args.pairs, control_tolerance=0.01)
    started = time.monotonic()
    try:
        if args.pairs < 6 or args.pairs % 6:
            raise ValueError('Use a multiple of six pairs for balanced arm orders')
        paths = dict(base=Path(args.baseline), candidate=Path(args.candidate or args.baseline),
                     reference=Path('reference.py'), checks=Path('checks.py'), benchmark=Path(__file__))
        report['sources'] = {k: dict(path=str(v), sha256=hashlib.sha256(v.read_bytes()).hexdigest())
                             for k, v in paths.items()}
        identical = report['sources']['base']['sha256'] == report['sources']['candidate']['sha256']
        report['aa_identical_sources'] = identical
        props = torch.cuda.get_device_properties(0)
        report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=props.name,
                                     multiprocessors=props.multi_processor_count,
                                     nvidia_smi=subprocess.check_output(['nvidia-smi', '--query-gpu=name,clocks.sm,clocks.mem,power.limit', '--format=csv'], text=True))
        reference = load(paths['reference'], 'reference')
        base = load(paths['base'], 'base')
        modules = {'base': base, 'candidate': base if identical else load(paths['candidate'], 'candidate')}
        for seed in map(int, args.weight_seeds.split(',')):
            weights = reference.generate_weights(seed)
            for pad_mib in map(int, args.pads_mib.split(',')):
                states = {}
                pad = torch.empty(pad_mib * 1024 * 1024, dtype=torch.uint8, device='cuda')
                screen = dict(weight_seed=seed, pad_mib=pad_mib, status='preparing', prepare_seconds={}, rows=[])
                report['screens'].append(screen)
                save(output, report)
                torch.cuda.reset_peak_memory_stats()
                for role in ('base', 'candidate'):
                    if role == 'candidate' and identical:
                        states[role] = states['base']
                        screen['prepare_seconds'][role] = 0.0
                        continue
                    before = time.monotonic()
                    states[role] = modules[role].prepare(weights)
                    torch.cuda.synchronize()
                    screen['prepare_seconds'][role] = time.monotonic() - before
                    print(json.dumps(dict(stage='prepared', role=role, seconds=screen['prepare_seconds'][role])), flush=True)
                    save(output, report)
                flush = torch.ones(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
                cases = [(m, layer, 'normal', True) for _, m in LOADS for layer in reference.LAYERS]
                if args.nearby:
                    cases += [(m, layer, dist, False) for m, layer, dist in NEARBY]
                for index, (m, layer, dist, scored) in enumerate(cases):
                    row = dict(m=m, layer=layer, dist=dist, scored=scored, status='running', records=[])
                    screen['rows'].append(row)
                    save(output, report)
                    run_case(reference, modules, states, weights, flush, row, args.pairs, seed + 100000 + index * 100)
                    save(output, report)
                    print(json.dumps({k: row[k] for k in ('m', 'layer', 'median_us', 'median_ratio')}), flush=True)
                summarize(screen)
                screen['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                screen['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
                save(output, report)
                states.clear()
                del flush, pad
                gc.collect()
                torch.cuda.empty_cache()
            del weights
            gc.collect()
            torch.cuda.empty_cache()
        report['geomean_ratio'] = {name: geomean(s['geomean_ratio'][name] for s in report['screens'])
                                   for name in ('candidate', 'base_repeat')}
        report['control_ok'] = all(s['control_ok'] for s in report['screens'])
        report['aa_ok'] = not identical or all(abs(l['ratio']['candidate'] - 1) <= 0.01
                                              for s in report['screens'] for l in s['loads'].values())
        report['status'] = 'passed' if report['control_ok'] and report['aa_ok'] else 'inconclusive'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        save(output, report)
        print(json.dumps({k: report.get(k) for k in ('status', 'geomean_ratio', 'control_ok', 'aa_ok', 'error')}), flush=True)
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
