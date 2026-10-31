"""External paired MoE screen; never a submission or an official score.

Run through the existing root-only launcher (all uploads use unique basenames):
  python3 work/modal/run_layer.py work/bench/screen_moe.py \
      best/moe.py work/candidates/moe_wgs_w4x2.py \
      outputs/openkernel-challenge/moe/reference.py \
      outputs/openkernel-challenge/moe/checks.py \
      outputs/openkernel-challenge/moe/nvfp4.py \
      --result screen-moe.json --args --baseline moe.py \
      --candidate moe_wgs_w4x2.py --output screen-moe.json

Default: one weight seed, twelve fresh-input pairs for each mandatory scored M.
Ratios are baseline / variant (greater than one is faster); take the median
paired ratio within each load, then the geomean across loads and weight seeds.
--weight-seeds 731283,731284 explicitly opts into multiple weight preparations.
H200 A/A calibration: candidate1.000724, control1.002585 (2026-10-30);
raw evidence: modal-lab/layer-okc-h200-20/screen-moe-aa-events.json.
--nearby adds unscored boundary loads and normal/empty/changed-input replay
checks on their existing graphs. Every timed replay is independently checked
against the public oracle, including the native baseline-repeat control.

A 1% control-geomean tolerance gates readiness, both overall and per weight
seed. Per-load control deviations are also reported. Identical source hashes
activate the same candidate-ratio gate for an A/A calibration. These are screen
readiness gates, not official correctness/performance certification. The
public oracle and preparing independent packed states are intentionally costly;
only graph replay is timed. Weights/states/graphs never survive a weight seed.
No module globals, extension names, kernel code, or compile flags are patched.
Candidates must supply their own collision-safe compiled-extension identities.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch


WORKSPACE = Path('/workspace')
LOADS = (('D1', 1), ('D16', 16), ('D32', 32), ('V1', 8), ('V16', 128), ('V32', 256))
NEARBY = (4, 17, 31, 33, 127, 129, 255, 257)
NAMES = ('base', 'candidate', 'base_repeat')
VARIANTS = ('candidate', 'base_repeat')
# Every consecutive group of three gives each implementation every position.
ORDERS = (('base', 'candidate', 'base_repeat'),
          ('candidate', 'base_repeat', 'base'),
          ('base_repeat', 'base', 'candidate'),
          ('base_repeat', 'candidate', 'base'),
          ('base', 'base_repeat', 'candidate'),
          ('candidate', 'base', 'base_repeat'))
FLUSH_BYTES = 256 * 1024 * 1024
REL_L2_TOL = 0.015
CONTROL_TOL = 0.01


def save(path, report):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_path(name):
    if Path(name).name != name or not name.endswith('.py'):
        raise ValueError('Source must be a Python basename in /workspace: ' + name)
    return WORKSPACE / name


def load(path, role):
    # Distinct even for renamed A/A copies; registration supports dataclasses and
    # ordinary module introspection without importing either source by basename.
    name = '_screen_moe_' + role
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError('Cannot load ' + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def capture(fn, data, state):
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(data, state)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(current)
    with torch.cuda.graph(graph, stream=stream):
        output = fn(data, state)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    # Instantiate/upload the captured graph before timing its first fresh input.
    # Eager warmup alone does not pay CUDA's first graph-replay setup cost.
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, output


def stable_input(data):
    return {key: value.clone() if isinstance(value, torch.Tensor) else dict(value)
            for key, value in data.items()}


def copy_input(target, fresh):
    if target['config'] != fresh['config']:
        raise ValueError('A captured graph cannot change its input configuration')
    for key, value in fresh.items():
        if isinstance(value, torch.Tensor):
            target[key].copy_(value)


def make_input(reference, m, seed, empty=False):
    data = reference.generate_input(bs=m, sharding='ep8', rank='busiest', seed=seed)
    if empty:
        # Rotate each token's distinct top-k IDs until none belong to the local
        # shard. Preserve public activations and route weights, including BF16.
        ids = data['topk_ids']
        count = data['config']['local_count']
        lo = data['config']['local_start']
        for _ in range(reference.N_EXPERTS // count):
            local_rows = ((ids >= lo) & (ids < lo + count)).any(dim=1)
            if not bool(local_rows.any()):
                break
            ids[local_rows] = (ids[local_rows] + count) % reference.N_EXPERTS
        # A token can occupy all eight shards; choose distinct remote IDs in
        # that case rather than relying on a rotation to find an empty shard.
        local_rows = ((ids >= lo) & (ids < lo + count)).any(dim=1)
        ids[local_rows] = (torch.arange(reference.ROUTED_PER_TOKEN, device=ids.device,
                                      dtype=ids.dtype) + lo + count) % reference.N_EXPERTS
    return data


def routing(data):
    cfg = data['config']
    counts = torch.bincount(data['topk_ids'].long().flatten(), minlength=cfg['n_experts'])
    counts = counts.cpu().tolist()
    local = counts[cfg['local_start']:cfg['local_start'] + cfg['local_count']]
    return dict(global_expert_histogram=counts, local_expert_histogram=local,
                shard_route_counts=[sum(counts[i:i + cfg['local_count']])
                                    for i in range(0, cfg['n_experts'], cfg['local_count'])],
                local_routes=sum(local), local_experts=sum(n > 0 for n in local),
                global_experts=sum(n > 0 for n in counts))


def summarize(records):
    ratios = {name: [record['paired_ratio'][name] for record in records] for name in VARIANTS}
    result = dict(median_us={name: statistics.median(r['us'][name] for r in records)
                            for name in NAMES}, paired_ratios=ratios,
                  median_ratio={name: statistics.median(values) for name, values in ratios.items()})
    result['control_within_tolerance'] = abs(result['median_ratio']['base_repeat'] - 1) <= CONTROL_TOL
    return result


def replay(reference, weights, fresh, captures, events, flush, order, record):
    record.update(status='checking_reference', routing=routing(fresh), order=list(order), us={}, correctness={})
    gold = reference.ref_kernel(fresh, weights)
    if not bool(torch.isfinite(gold).all()):
        raise RuntimeError('Public reference contains NaN or Inf')
    for data, _, _ in captures.values():
        copy_input(data, fresh)
    for name in order:
        _, graph, output = captures[name]
        # The read-only flush and input copies are ordered before the start
        # event. Nothing except native kernel_fn's captured work lies between
        # timing events; correctness checks never enter the measured graph.
        sink = flush.sum()
        begin, end = events[name]
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = begin.elapsed_time(end) * 1000.0
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError('Invalid CUDA timing for ' + name + ': ' + str(elapsed))
        record['us'][name] = elapsed
        del sink
    # Check every arm after the complete timed triplet, so reference work cannot
    # change the GPU's activity/clocks between arms. Each graph owns its output.
    for name in order:
        output = captures[name][2]
        # Direct compare also supports deliberate empty routes.
        passed, message = reference.compare(output, gold, REL_L2_TOL)
        record['correctness'][name] = dict(passed=bool(passed), message=str(message))
        if not passed:
            raise RuntimeError(name + ' failed public reference: ' + str(message))
    record['paired_ratio'] = {name: record['us']['base'] / record['us'][name] for name in VARIANTS}
    record['status'] = 'passed'


def run_case(reference, modules, weights, states, flush, args, row, case_index, seed_index):
    captures, events = {}, {}
    m = row['m']
    input_seed = args.seed + m * 10000
    initial = make_input(reference, m, input_seed)
    try:
        row['status'] = 'capturing'
        for name in NAMES:
            role = 'candidate' if name == 'candidate' else 'base'
            data = stable_input(initial)
            graph, output = capture(modules[role].kernel_fn, data, states[role])
            captures[name] = (data, graph, output)
            events[name] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            # CUDA events initialize lazily on first record. Pay that host-side
            # setup before any start event can include an idle GPU submission gap.
            begin, end = events[name]
            begin.record()
            end.record()
            end.synchronize()
        # Keep no loop-local aliases alive after clearing the captures below.
        del data, graph, output, initial
        row['status'] = 'timing'
        for pair in range(args.pairs):
            seed = input_seed + pair + 1
            fresh = make_input(reference, m, seed)
            record = dict(pair=pair, input_seed=seed, input_kind='public_busiest')
            row['records'].append(record)
            order = ORDERS[(pair + 3 * ((case_index + seed_index) % 2)) % len(ORDERS)]
            replay(reference, weights, fresh, captures, events, flush, order, record)
            del fresh
        row.update(summarize(row['records']))
        if not row['scored']:
            row['replay_checks'] = []
            # Reuse the very same captured addresses across nonempty -> empty
            # -> different nonempty -> empty -> original nonempty transitions.
            # This also exercises repeated input A after intervening input B.
            sequence = ((False, 101), (True, 102), (False, 103),
                        (True, 104), (False, 101), (True, 102))
            for index, (empty, offset) in enumerate(sequence):
                seed = input_seed + args.pairs + offset
                fresh = make_input(reference, m, seed, empty=empty)
                record = dict(replay=index, input_seed=seed, input_kind='empty' if empty else 'public_busiest')
                row['replay_checks'].append(record)
                replay(reference, weights, fresh, captures, events, flush, ORDERS[index], record)
                del fresh
            row['replay_summary'] = summarize(row['replay_checks'])
        row['status'] = 'passed'
    except BaseException as exc:
        row.update(status='failed', error=repr(exc))
        raise
    finally:
        captures.clear()
        events.clear()
        gc.collect()


def aggregate(rows):
    return {name: geomean(row['median_ratio'][name] for row in rows) for name in VARIANTS}


def run_seed(reference, modules, args, seed, seed_index, entry, report, output_path):
    weights, states, flush = None, {}, None
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        entry['status'] = 'generating_weights'
        before = time.monotonic()
        weights = reference.generate_weights(seed)
        torch.cuda.synchronize()
        entry['generate_weights_seconds'] = time.monotonic() - before
        entry['prepare_seconds'] = {}
        for name in ('base', 'candidate'):
            entry['status'] = 'preparing_' + name
            save(output_path, report)
            before = time.monotonic()
            states[name] = modules[name].prepare(weights)
            torch.cuda.synchronize()
            entry['prepare_seconds'][name] = time.monotonic() - before
        flush = torch.ones(FLUSH_BYTES // 4, device='cuda', dtype=torch.float32)
        cases = [(name, m, True) for name, m in LOADS]
        if args.nearby:
            cases.extend(('M' + str(m), m, False) for m in NEARBY)
        for index, (name, m, scored) in enumerate(cases):
            row = dict(load=name, m=m, scored=scored, status='pending', records=[])
            entry['scored' if scored else 'nearby'].append(row)
            save(output_path, report)
            run_case(reference, modules, weights, states, flush, args, row, index, seed_index)
            save(output_path, report)
            print(json.dumps(dict(weight_seed=seed, load=name, scored=scored,
                                  median_us=row['median_us'], median_ratio=row['median_ratio'])), flush=True)
        if {(r['load'], r['m']) for r in entry['scored']} != set(LOADS):
            raise RuntimeError('Missing mandatory scored loads')
        if any(len(r['records']) != args.pairs or len(r['records']) < 3 or r['status'] != 'passed'
               for r in entry['scored']):
            raise RuntimeError('Incomplete scored pairs or failed correctness')
        entry['geomean_ratio'] = aggregate(entry['scored'])
        entry['control_within_tolerance'] = abs(entry['geomean_ratio']['base_repeat'] - 1) <= CONTROL_TOL
        entry['status'] = 'passed'
    except BaseException as exc:
        entry.update(status='failed', error=repr(exc))
        raise
    finally:
        entry.update(elapsed_seconds=time.monotonic() - started,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        # All per-case graphs have left scope. Only scalar/list observations
        # remain in the report; never retain weights across seeds.
        states.clear()
        weights = flush = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', default='screen-moe.json')
    parser.add_argument('--seed', type=int, default=731283)
    parser.add_argument('--pairs', type=int, default=12)
    parser.add_argument('--weight-seeds', help='Comma-separated public weight seeds; default: --seed only')
    parser.add_argument('--nearby', action='store_true')
    args = parser.parse_args()
    output_path = WORKSPACE / args.output
    report = dict(kind='screen_moe_paired', official_score=False, status='running', errors=[],
                  seed=args.seed, pairs_per_case=args.pairs, nearby_requested=args.nearby,
                  required_loads=[dict(load=name, m=m) for name, m in LOADS],
                  cache_protocol='256 MiB read-only sum before every timed graph replay',
                  timing_protocol='CUDA events around one native kernel_fn graph replay only',
                  aggregation='median baseline/variant paired ratio per load, then geomean across loads/seeds; scored only',
                  reference_relative_l2_tolerance=REL_L2_TOL, control_ratio_tolerance=CONTROL_TOL,
                  gating=dict(ready=False, reasons=['screen not complete']), weight_screens=[])
    exit_code = 1
    try:
        save(output_path, report)
        if args.pairs < 3:
            raise ValueError('--pairs must be at least 3; twelve is the balanced default')
        seeds = [args.seed] if args.weight_seeds is None else [int(s.strip()) for s in args.weight_seeds.split(',')]
        if not seeds:
            raise ValueError('--weight-seeds must contain at least one integer')
        report['weight_seeds'] = seeds
        paths = {'base': source_path(args.baseline), 'candidate': source_path(args.candidate),
                 'reference': WORKSPACE / 'reference.py', 'checks': WORKSPACE / 'checks.py',
                 'nvfp4': WORKSPACE / 'nvfp4.py', 'benchmark': Path(__file__)}
        report['sources'] = {name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                             for name, path in paths.items()}
        report['aa_identical_sources'] = report['sources']['base']['sha256'] == report['sources']['candidate']['sha256']
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        report['environment'] = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                                     cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                     gpu=props.name, capability=list(torch.cuda.get_device_capability()),
                                     gpu_memory_bytes=props.total_memory, multiprocessors=props.multi_processor_count,
                                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                     float32_matmul_precision=torch.get_float32_matmul_precision(),
                                     environment={key: os.environ.get(key) for key in
                                                  ('CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING', 'TORCH_CUDA_ARCH_LIST')})
        reference = load(paths['reference'], 'reference')
        modules = {name: load(paths[name], name) for name in ('base', 'candidate')}
        for index, seed in enumerate(seeds):
            entry = dict(weight_seed=seed, status='pending', scored=[], nearby=[])
            report['weight_screens'].append(entry)
            run_seed(reference, modules, args, seed, index, entry, report, output_path)
            save(output_path, report)
        screens = report['weight_screens']
        if len(screens) != len(seeds) or any(s['status'] != 'passed' for s in screens):
            raise RuntimeError('Incomplete weight screens')
        report['loads'] = {name: dict(m=m, geomean_ratio={variant: geomean(
            next(r for r in screen['scored'] if r['load'] == name)['median_ratio'][variant]
            for screen in screens) for variant in VARIANTS}) for name, m in LOADS}
        report['geomean_ratio'] = {name: geomean(s['geomean_ratio'][name] for s in screens) for name in VARIANTS}
        control_ok = (abs(report['geomean_ratio']['base_repeat'] - 1) <= CONTROL_TOL
                      and all(s['control_within_tolerance'] for s in screens))
        aa_ok = not report['aa_identical_sources'] or (
            abs(report['geomean_ratio']['candidate'] - 1) <= CONTROL_TOL
            and all(abs(s['geomean_ratio']['candidate'] - 1) <= CONTROL_TOL for s in screens))
        reasons = []
        if not control_ok:
            reasons.append('baseline-repeat geomean outside 1% tolerance overall or for a weight seed')
        if not aa_ok:
            reasons.append('identical-source A/A candidate geomean outside 1% tolerance')
        report['gating'] = dict(ready=control_ok and aa_ok, all_six_loads_complete=True,
                                minimum_three_pairs=True, all_replays_reference_passed=True,
                                nearby_complete=args.nearby, control_passed=control_ok,
                                aa_calibration_passed=aa_ok if report['aa_identical_sources'] else None,
                                reasons=reasons, scope='external screen only; official gate still required')
        report['status'] = 'passed' if report['gating']['ready'] else 'inconclusive'
        exit_code = 0
    except BaseException as exc:
        report['status'] = 'failed'
        report['errors'].append(dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        report['gating'] = dict(ready=False, reasons=['screen failed; see errors and per-case records'])
    finally:
        screens = report['weight_screens']
        for key in ('peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
            report[key] = max((s.get(key, 0) for s in screens), default=0)
        save(output_path, report)
        print(json.dumps(dict(status=report['status'], gating=report['gating'],
                              geomean_ratio=report.get('geomean_ratio'), result=str(output_path))), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
