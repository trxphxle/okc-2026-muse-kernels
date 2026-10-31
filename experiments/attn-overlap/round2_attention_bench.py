"""Lane-local attention screen and native ceiling baseline, never a submission.

Uploads: this script, decode_current.py (or attention candidate),
compare_decode_public.py, round2_gather.py, attention/reference.py and attention/checks.py.
The embedded decoder is invoked only through _launch7; no heavy layer prepare.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import torch
import reference
import compare_decode_public as bench
from round2_gather import native_batch

bench.torch = torch
LOADS = (('D1', 1, 1), ('D16', 16, 1), ('D32', 32, 1),
         ('V1', 1, 8), ('V16', 16, 8), ('V32', 32, 8))
WEIGHTS = {'sliding': 46, 'full': 16}


def attention_fn(module):
    if hasattr(module, '_launch7'):
        return lambda data, state: module._launch7(data, pdl=False)
    return lambda data, state: module.kernel_fn(data)


def oracle(data, output):
    raw = reference.collect(data, output)
    passed, message = reference.check_implementation(data, raw)
    return dict(passed=bool(passed), message=str(message))


def compare(a, b):
    a, b = a.float(), b.float()
    diff = a - b
    norms = b.norm(dim=-1)
    return dict(finite=bool(torch.isfinite(a).all()),
                l2=float(diff.norm() / b.norm().clamp_min(1e-12)),
                row_l2=float((diff.norm(dim=-1) /
                             norms.clamp_min(float(norms.mean()) * 0.1).clamp_min(1e-12)).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='decode_current.py')
    parser.add_argument('--candidate', default=None)
    parser.add_argument('--output', default='round2-baseline.json')
    parser.add_argument('--seed', type=int, default=919233)
    parser.add_argument('--pairs', type=int, default=21)
    parser.add_argument('--profiles', type=int, default=3)
    parser.add_argument('--inputs', type=int, default=1)
    parser.add_argument('--loads', default=','.join(x[0] for x in LOADS))
    parser.add_argument('--layers', default='sliding,full')
    parser.add_argument('--pad-mib', type=int, default=0)
    args = parser.parse_args()
    args.candidate = args.candidate or args.baseline
    report = dict(kind='round2_attention_paired', official_score=False,
                  seed=args.seed, pairs=args.pairs, inputs=args.inputs, pad_mib=args.pad_mib,
                  source_sha256={x: bench.sha(x) for x in {args.baseline, args.candidate}},
                  environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name()),
                  timing_protocol='256MiB sum inside graph before captured external events; native profiles separate',
                  shape_distribution='attention-board full coherent / sliding soft',
                  results=[], loads={}, artifact_files=[], status='running')
    output = Path(args.output)
    start = time.monotonic()

    def save():
        bench.save(output, report)

    save()
    try:
        pad = torch.empty(args.pad_mib * 1024 * 1024, dtype=torch.uint8, device='cuda')
        base = bench.load(args.baseline, 'round2_base')
        candidate = bench.load(args.candidate, 'round2_candidate')
        functions = {'base': attention_fn(base), 'candidate': attention_fn(candidate)}
        functions['base_repeat'] = functions['base']
        flush = torch.ones(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
        for load, batch, length in LOADS:
            if load not in args.loads.split(','):
                continue
            for layer in args.layers.split(','):
                shape = dict(batch=batch, q_len=length, ctx=80000, heads='tp8', layer=layer, page=1,
                             spread='agentic' if batch > 1 else 'fixed',
                             dist='coherent' if layer == 'full' else 'soft')
                row = dict(load=load, layer=layer, shape=shape, inputs=[])
                report['results'].append(row)
                for input_index in range(args.inputs):
                    seed = args.seed + batch * 100 + length * 1000 + (layer == 'full') * 10000 + input_index
                    source = reference.generate_input(seed=seed, **shape)
                    data = {'base': bench.clone_data(source), 'candidate': bench.clone_data(source)}
                    data['base_repeat'] = data['base']
                    del source
                    captured = {name: bench.capture(fn, data[name], None, flush)
                                for name, fn in functions.items()}
                    record = dict(seed=seed, seq_lens=data['base']['seq_lens'].cpu().tolist(), samples=[])
                    row['inputs'].append(record)
                    names = tuple(functions)
                    for pair in range(args.pairs):
                        rotated = names[pair % len(names):] + names[:pair % len(names)]
                        order = rotated if pair % 2 == 0 else rotated[::-1]
                        times = {name: bench.measure(captured[name]) for name in order}
                        record['samples'].append(dict(pair=pair, order=order, us=times))
                    record['median_us'] = {n: statistics.median(x['us'][n] for x in record['samples'])
                                           for n in names}
                    record['comparisons'] = {name: compare(captured[name]['output'], captured['base']['output'])
                                             for name in ('candidate', 'base_repeat')}
                    record['oracle'] = {name: oracle(data[name], captured[name]['output']) for name in names}
                    if not all(x['passed'] for x in record['oracle'].values()):
                        raise RuntimeError(f'Attention oracle rejected {load}/{layer}')
                    record['native'] = {}
                    if input_index == 0 and args.profiles:
                        native_captures = {name: bench.capture(functions[name], data[name], None)
                                           for name in ('base', 'candidate', 'base_repeat')}
                        native_orders = []
                        for repeat in range(args.profiles):
                            order = ['base', 'candidate', 'base_repeat']
                            shift = repeat % len(order)
                            order = order[shift:] + order[:shift]
                            native_orders.append(order if repeat % 2 == 0 else order[::-1])
                        filename = output.with_name(f'{output.stem}-{load}-{layer}-native-batch.json')
                        record['native'] = native_batch(bench, native_captures, flush, native_orders,
                                                        filename, report, output)
                        del native_captures
                        save()
                    del captured, data
                    gc.collect()
                    save()
                row['median_us'] = {name: statistics.median(i['median_us'][name] for i in row['inputs'])
                                     for name in functions}
                row['ratio'] = {name: statistics.median(sample['us']['base'] / sample['us'][name]
                                                       for i in row['inputs'] for sample in i['samples'])
                                for name in ('candidate', 'base_repeat')}
                row['native_median_us'] = {name: statistics.median(t['launches'][0]['duration_us']
                                                                  for i in row['inputs']
                                                                  for t in i['native'].get(name, []))
                                            for name in ('base', 'candidate')} if args.profiles else {}
                print(json.dumps(dict(load=load, layer=layer, median_us=row['median_us'],
                                      ratio=row['ratio'], native_median_us=row['native_median_us'])), flush=True)
                save()
        for load, _, _ in LOADS:
            rows = [r for r in report['results'] if r['load'] == load]
            if len(rows) != 2:
                continue
            weighted = {name: sum(WEIGHTS[r['layer']] * r['median_us'][name] for r in rows) / 62
                        for name in functions}
            report['loads'][load] = dict(weighted_us=weighted,
                ratio={name: weighted['base'] / weighted[name] for name in ('candidate', 'base_repeat')},
                native_weighted_us={name: sum(WEIGHTS[r['layer']] * r['native_median_us'][name]
                                             for r in rows) / 62 for name in ('base', 'candidate')}
                if args.profiles else {})
        report['geomean_ratio'] = {name: math.exp(statistics.mean(math.log(x['ratio'][name])
                                                                 for x in report['loads'].values()))
                                   for name in ('candidate', 'base_repeat')} if report['loads'] else {}
        report['aa_accepted'] = all(abs(x['ratio']['base_repeat'] - 1) <= 0.01 for x in report['loads'].values())
        report['status'] = 'passed'
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        save()
        print(json.dumps(dict(status=report['status'], result=str(output),
                              geomean_ratio=report.get('geomean_ratio'),
                              aa_accepted=report.get('aa_accepted'))), flush=True)


if __name__ == '__main__':
    main()
