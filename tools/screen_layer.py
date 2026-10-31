"""Fast paired screen: candidate vs baseline whole layer; not an official score.

Validated on H200 with identical baseline/candidate copies (2026-10-30):
candidate ratio 0.998685, baseline-repeat 1.001234. Repeat this A/A after timing
protocol changes; require both geomean ratios within 1% of 1.00.

Run on Modal through the existing launcher (uploads are basenames in /workspace):
  python3 work/modal/run_layer.py work/bench/screen_layer.py \
      best/decode_layer.py path/to/candidate.py \
      outputs/openkernel-challenge/decode-layer/reference.py \
      outputs/openkernel-challenge/decode-layer/checks.py \
      outputs/openkernel-challenge/decode-layer/nvfp4.py \
      --result screen-result.json \
      --args --baseline decode_layer.py --candidate candidate.py

Screen mode by default: one weight seed, 3 pairs per shape, reference check on
the first pair and on any disagreement. Ratio > 1 means faster than baseline.

Both files load into one process. If the candidate changes CUDA source, give its
load_inline extensions names that differ from the baseline's, or the second
build can collide with the first.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path
import torch
import reference

LOADS = (('D1', 1, 1), ('D16', 16, 1), ('D32', 32, 1), ('V1', 1, 8), ('V16', 16, 8), ('V32', 32, 8))
LAYER_WEIGHT = {'sliding': 46, 'full': 16}

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True)
parser.add_argument('--candidate', required=True)
parser.add_argument('--seed', type=int, default=731283)
parser.add_argument('--pairs', type=int, default=3)
parser.add_argument('--loads', default=','.join(n for n, _, _ in LOADS))
parser.add_argument('--shared-state', action='store_true',
                    help='prepare once with the candidate (only if both use the same state layout)')
parser.add_argument('--output', default='screen-result.json')
args = parser.parse_args()

def load(name):
    path = Path('/workspace') / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

RESULT = Path('/workspace') / args.output
report = dict(kind='screen_whole_layer_paired', official_score=False, seed=args.seed,
              pairs_per_case=args.pairs, torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(), shared_state=args.shared_state,
              cache_protocol='256 MiB read-only sum before each timed graph',
              source_sha256={n: hashlib.sha256((Path('/workspace') / n).read_bytes()).hexdigest()
                             for n in (args.baseline, args.candidate)},
              results=[], loads={})

def save():
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

def capture(fn, data, state):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(data, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn(data, state)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph, output

def errors(actual, expected, data, weights, require_reference=False):
    ids_equal = bool(torch.equal(actual[2], expected[2]))
    result = {'ids_equal': ids_equal}
    require_reference = require_reference or not ids_equal
    for key, a, b in zip(('hidden', 'stream'), actual[:2], expected[:2]):
        if not torch.isfinite(a).all():
            raise RuntimeError('Nonfinite ' + key)
        a, b = a.float(), b.float()
        delta = a - b
        rel = float(delta.norm() / b.norm().clamp_min(1e-12))
        norms = b.norm(dim=-1)
        row = float((delta.norm(dim=-1) / norms.clamp_min(float(norms.mean()) * 0.1).clamp_min(1e-12)).max())
        if rel > 0.003 or row > 0.015:
            require_reference = True
        result[key + '_l2'] = rel
        result[key + '_row_l2'] = row
    if require_reference:
        # Baseline agreement cannot adjudicate nondeterministic reductions; the
        # public oracle checks residual updates, rows, caches and routing.
        passed, message = reference.check_implementation(data, reference.collect(data, actual), weights)
        result.update(reference_passed=passed, reference_message=message)
        if not passed:
            raise RuntimeError('Independent reference rejected output: ' + message)
    return result

try:
    base, candidate = load(args.baseline), load(args.candidate)
    t0 = time.monotonic()
    weights = reference.generate_weights(args.seed)
    states = {'candidate': candidate.prepare(weights)}
    report['prepare_seconds_candidate'] = time.monotonic() - t0
    if args.shared_state:
        states['base'] = states['candidate']
    else:
        t0 = time.monotonic()
        states['base'] = base.prepare(weights)
        report['prepare_seconds_base'] = time.monotonic() - t0
    flush = torch.ones(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
    save()
    fns = {'base': (base.kernel_fn, 'base'), 'candidate': (candidate.kernel_fn, 'candidate'),
           'base_repeat': (base.kernel_fn, 'base')}
    wanted = set(args.loads.split(','))
    for load_name, batch, length in LOADS:
        if load_name not in wanted:
            continue
        for layer in ('sliding', 'full'):
            shape = dict(batch=batch, q_len=length, ctx=80000, layer=layer, page=1,
                         spread='agentic' if batch > 1 else 'fixed', dist='soft')
            seed = 919233 + batch * 100 + length * 1000 + (layer == 'full') * 10000
            data = reference.generate_input(seed=seed, **shape)
            graphs, outputs = {}, {}
            for name, (fn, state_name) in fns.items():
                graphs[name], outputs[name] = capture(fn, data, states[state_name])
            events = {n: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for n in graphs}
            row = dict(load=load_name, batch=batch, q_len=length, layer=layer, records=[])
            for pair in range(args.pairs):
                fresh = reference.generate_input(seed=seed + pair + 1, **shape)
                # Target the busiest EP8 shard using the baseline's routing, outside timing.
                eager = base.kernel_fn(fresh, states['base'])
                counts = torch.bincount((eager[2].long() // 32).flatten(), minlength=8)
                fresh['local_start'].fill_(32 * int(counts.argmax()))
                for key, value in fresh.items():
                    if isinstance(value, torch.Tensor):
                        data[key].copy_(value)
                times = {}
                order = ('base', 'candidate', 'base_repeat') if pair % 2 == 0 else ('base_repeat', 'candidate', 'base')
                for name in order:
                    flush.sum()
                    begin, end = events[name]
                    begin.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    times[name] = begin.elapsed_time(end) * 1000.0
                comparisons = {n: errors(outputs[n], outputs['base'], data, weights, pair == 0)
                               for n in ('base_repeat', 'candidate')}
                row['records'].append(dict(pair=pair, seed=seed + pair + 1, us=times, errors=comparisons))
                del fresh, eager
            row['median_us'] = {n: statistics.median(r['us'][n] for r in row['records']) for n in graphs}
            report['results'].append(row)
            save()
            print(json.dumps(dict(load=load_name, layer=layer, median_us=row['median_us'])), flush=True)
            del data, graphs, outputs, events
            gc.collect()
    # Official aggregation: per load, 46:16 weighted sum of sliding/full; then
    # the geometric mean over loads of baseline time / variant time.
    logs = {'candidate': [], 'base_repeat': []}
    for load_name, _, _ in LOADS:
        rows = [r for r in report['results'] if r['load'] == load_name]
        if len(rows) != 2:
            continue
        total = {n: sum(LAYER_WEIGHT[r['layer']] * r['median_us'][n] for r in rows) / 62 for n in fns}
        ratios = {n: total['base'] / total[n] for n in logs}
        report['loads'][load_name] = dict(weighted_us=total, ratio=ratios)
        for n in logs:
            logs[n].append(math.log(ratios[n]))
    report['geomean_ratio'] = {n: math.exp(sum(v) / len(v)) for n, v in logs.items() if v}
    report['status'] = 'passed'
    report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
except BaseException as exc:
    report.update(status='failed', error=repr(exc))
    raise
finally:
    save()
    print(json.dumps(dict(status=report.get('status'), geomean_ratio=report.get('geomean_ratio'),
                          result=str(RESULT))), flush=True)
