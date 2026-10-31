#!/usr/bin/env python3
"""Diagnostic, NOT a submission or a measured full-kernel optimization.

Upload this file, layout_bound.cu, muse_moe_w4a16.py and the public MoE
reference.py/checks.py/nvfp4.py to /workspace. Run:
  python layout_bound.py --baseline muse_moe_w4a16.py --output layout_bound
Optional --full-kernel-us baseline_times.json accepts measured D32/V16/V32
microseconds (a flat object). Unchanged D1/D16/V1 have zero credited savings.

Uses the upstream prepare() WGS packers without compiling any native kernel or
building duplicate full states. Exact native WGS per-CTA A/B address sequences
are reproduced, including repeated expert jobs and phase-A trailing-share
reorder. Only the weight transfer is modeled: x, MMA, unit epilogues, routing,
fixups and cross-CTA dependencies are absent. Timed transfers wait for every
full volatile TMA transaction and publish completed bytes/address checksums.
A separate full-byte checksum consumer validates the same sequence before and
after timing. Savings are conditional layout-port estimates, not real wins.
No physical DRAM channel mapping is inferred from virtual allocation offsets.

Attribution: upstream WGS packing/scheduling/TMA policy by alygg77 and vjeux,
Apache-2.0; upstream file is imported unchanged, never copied into submissions.
Only this diagnostic's small extension is built. Hard process deadline defaults
to 840 seconds, including compilation; partial JSON survives timeout/failure.
"""
import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback

STAGE = 36864
GRID = 132
OFFSETS = (0, 512, 2 * 1024 * 1024 + 4096, 31 * 1024 * 1024 + 512)
LOADS = {'D32': 32, 'V16': 128, 'V32': 256}
ARMS = ('current', 'interleaved', 'repeat')


def save(path, report):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def schedule(counts, costs):
    # route(): fat jobs first, expert order within fat and skinny groups.
    jobs = [(e, min(64, n-q)) for e, n in enumerate(counts) if n > 8
            for q in range(0, n, 64)]
    jobs += [(e, n) for e, n in enumerate(counts) if 0 < n <= 8]
    per_cta = [[] for _ in range(GRID)]
    ranges = [[] for _ in range(GRID)]
    for pb, units in enumerate((32, 16)):
        nstage = units * 16
        cs = [costs[pb][(n+7)//8-1] for _, n in jobs]
        prefix = [0]
        for cost in cs:
            prefix.append(prefix[-1] + cost*nstage)
        def at(x):
            if not jobs:
                return 0
            j = bisect.bisect_right(prefix, x)-1
            return j*nstage + (x-prefix[j])//cs[j]
        for c in range(GRID):
            lo = at(prefix[-1]*c//GRID)
            hi = len(jobs)*nstage if c == GRID-1 else at(prefix[-1]*(c+1)//GRID)
            end_unit = ((hi-1)//16)*16
            split = end_unit if pb == 0 and hi%16 and end_unit > lo else lo
            ranges[c].append([lo, hi, split])
            for gs in list(range(split, hi)) + list(range(lo, split)):
                j, stage = divmod(gs, nstage)
                e = jobs[j][0]
                # Canonical flattened expert-major stage ID, A followed by B.
                per_cta[c].append(pb*32*512 + e*nstage + stage)
    flat = [s for c in per_cta for s in c]
    bounds = [0]
    for c in per_cta:
        bounds.append(bounds[-1]+len(c))
    assert len(flat) == len(jobs)*768
    return jobs, per_cta, flat, bounds, ranges


def capture(torch, fn):
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    current.wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(current)
    with torch.cuda.graph(graph, stream=stream):
        fn()
    current.wait_stream(stream)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def packed_weights(torch, baseline, weights):
    # Exactly prepare() lines building w[12] and w[13], expert at a time.
    dev = weights['w13'].device
    rb = torch.arange(512, device=dev)
    g = torch.arange(8, device=dev)
    rows_a = torch.stack((8*rb[:, None]+g, 4096+8*rb[:, None]+g), 1).reshape(-1)
    rows_a = baseline._kperm_rows(rows_a)
    rows_b = torch.arange(4096, device=dev)
    packed = torch.empty((32*768, STAGE), dtype=torch.uint8, device=dev)
    for pb, (key, rows, stages) in enumerate((('w13', rows_a, 512), ('w2', rows_b, 256))):
        for e in range(32):
            start = pb*32*512+e*stages
            packed[start:start+stages].copy_(baseline._wgs_copy(weights[key][e], weights[key+'_sf'][e], rows).view(stages, STAGE))
    return packed


def validate_layout(torch, packed, view, interleaved, stage_checksums):
    # Full bytewise inverse-layout equality in bounded expert-sized chunks;
    # additionally checks both checksum components for every source byte.
    checks = []
    for pb, stages in enumerate((512, 256)):
        base = pb*32*512
        part = view[base:base+32*stages]
        for e in range(32):
            actual = part.view(stages, 32, STAGE)[:, e] if interleaved else part[e*stages:(e+1)*stages]
            expected = packed[base+e*stages:base+(e+1)*stages]
            if not torch.equal(actual, expected):
                raise RuntimeError('Full-byte repack/unpack equality failed')
            a = actual.to(torch.int64).view(stages, 288, 128)
            pos = torch.arange(1, STAGE+1, device='cuda', dtype=torch.int64).view(1, 288, 128)
            got = torch.stack((a.sum(1), (a*pos).sum(1)), -1)
            if not torch.equal(got, stage_checksums[base+e*stages:base+(e+1)*stages]):
                raise RuntimeError('Full-byte layout checksum failed')
            checks.append([int(got[..., 0].sum()), int(got[..., 1].sum())])
    return {'full_bytewise_unpack_equal': True, 'every_byte_checksum_equal': True,
            'expert_phase_checksums': checks}


def worker(args, output):
    report = {'status': 'starting', 'diagnostic_only': True, 'cases': [], 'errors': [],
              'weight_seed': args.weight_seed, 'route_seeds': args.route_seeds,
              'offsets_bytes': list(OFFSETS), 'reps': args.reps,
              'scope': __doc__, 'effective_bytes_are': 'TMA requested weight bytes including repeated jobs; not measured DRAM traffic',
              'board': {'weights': 'six equally weighted loads', 'status': 'pending'}}
    save(output, report)
    started = time.monotonic()
    try:
        import torch
        from torch.utils.cpp_extension import load as cuda_load
        root = Path(__file__).resolve().parent
        baseline_path = Path(args.baseline).resolve()
        reference_path = baseline_path.parent/'reference.py'
        sys.path.insert(0, str(baseline_path.parent))
        report['sources_sha256'] = {str(p): digest(p) for p in
            (Path(__file__), root/'layout_bound.cu', baseline_path, reference_path,
             baseline_path.parent/'checks.py', baseline_path.parent/'nvfp4.py')}
        baseline = load(baseline_path, '_layout_baseline')
        reference = load(reference_path, '_layout_reference')
        source = baseline_path.read_text()
        costs = []
        for tag in ('OKC_WGSA', 'OKC_WGSB'):
            match = re.search(r'^#define '+tag+r' ([0-9, ]+)', source, re.M)
            costs.append([int(v) for v in match.group(1).split(',')])
        for token in ('#define OKC_SKT 8', '#define OKC_FT_T 64', '#define OKC_WGSREORD 1', 'constexpr int WGS_NS = 3'):
            if token not in source:
                raise ValueError('Unsupported upstream scheduling contract: '+token)
        assert baseline.LOCAL_START == 0 and baseline.LOCAL_EXPERTS == 32 and baseline.HIDDEN == 4096
        assert '-DOKC_WGS=1' in baseline.MAIN_DEFINES
        props = torch.cuda.get_device_properties(0)
        if props.major != 9 or props.multi_processor_count != GRID:
            raise ValueError('Probe requires SM90 H200-class device with exactly 132 SMs')
        report['device'] = {'name': props.name, 'sms': props.multi_processor_count,
                            'total_memory': props.total_memory, 'torch': torch.__version__, 'cuda': torch.version.cuda}
        report['cost_tables'] = costs
        report['status'] = 'generating_weights'
        save(output, report)
        weights = reference.generate_weights(args.weight_seed)
        if not baseline._scales_prmt_exact(weights['w13_sf'][:32], weights['w2_sf'][:32]):
            raise ValueError('Public weights do not select native WGS: PRMT scale check failed')
        packed = packed_weights(torch, baseline, weights)
        del weights
        torch.cuda.synchronize()
        report['packing_seconds'] = time.monotonic()-started
        # Small per-stage/per-lane checksum reference; avoid full-weight int64 expansion.
        stage_checksums = torch.empty((packed.shape[0], 128, 2), dtype=torch.int64, device='cuda')
        pos = torch.arange(1, STAGE+1, device='cuda', dtype=torch.int64).view(1, 288, 128)
        for start in range(0, packed.shape[0], 64):
            x = packed[start:start+64].to(torch.int64).view(-1, 288, 128)
            stage_checksums[start:start+64, :, 0] = x.sum(1)
            stage_checksums[start:start+64, :, 1] = (x*pos).sum(1)
        del x
        report['packed_checksums'] = [int(stage_checksums[..., k].sum()) for k in (0, 1)]
        report['status'] = 'compiling_probe_only'
        save(output, report)
        os.environ.setdefault('MAX_JOBS', '2')
        ext = cuda_load(name='layout_bound_'+digest(root/'layout_bound.cu')[:12],
                        sources=[str(root/'layout_bound.cu')],
                        extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a'], verbose=False)
        flush = torch.ones(256*1024*1024//4, dtype=torch.int32, device='cuda')
        # Warm two fixed backing allocations once, then vary both views' offsets.
        size = packed.numel()
        backing = {name: torch.empty(size+max(OFFSETS), dtype=torch.uint8, device='cuda')
                   for name in ('current', 'interleaved')}
        for tensor in backing.values():
            tensor.zero_()
        torch.cuda.synchronize()
        report['allocations'] = {k: {'base_ptr': v.data_ptr(), 'bytes': v.numel()} for k, v in backing.items()}
        for offset_index, offset in enumerate(OFFSETS):
            views = {k: v[offset:offset+size].view_as(packed) for k, v in backing.items()}
            views['current'].copy_(packed)
            for pb, stages in enumerate((512, 256)):
                base = pb*32*512
                views['interleaved'][base:base+32*stages].view(stages, 32, STAGE).copy_(
                    packed[base:base+32*stages].view(32, stages, STAGE).permute(1, 0, 2))
            correctness_before = {k: validate_layout(torch, packed, v, k=='interleaved', stage_checksums) for k, v in views.items()}
            for label, m in LOADS.items():
                for seed in args.route_seeds:
                    data = reference.generate_input(bs=m, sharding='ep8', rank='busiest', seed=seed)
                    cfg = data['config']
                    assert cfg['local_start']==0 and cfg['local_count']==32
                    counts = torch.bincount(data['topk_ids'].long().flatten(), minlength=256).cpu().tolist()[:32]
                    jobs, streams, flat, bounds, ranges = schedule(counts, costs)
                    del data
                    row = {'load': label, 'm': m, 'seed': seed, 'offset_bytes': offset,
                           'counts': counts, 'jobs': jobs, 'ranges_a_b_lo_hi_split': ranges,
                           'cta_stage_counts': [len(s) for s in streams], 'stage_count': len(flat),
                           'logical_weight_bytes': len(flat)*STAGE, 'unique_weight_bytes': len(set(flat))*STAGE,
                           'sequence_sha256': hashlib.sha256(json.dumps(streams).encode()).hexdigest(),
                           'correctness_before': correctness_before, 'records': [], 'status': 'capturing',
                           'data_ptrs': {k: v.data_ptr() for k, v in views.items()}}
                    report['cases'].append(row)
                    save(output, report)
                    ids = torch.tensor(flat, dtype=torch.int64, device='cuda')
                    inter = []
                    for s in flat:
                        pb = int(s >= 32*512)
                        n = 256 if pb else 512
                        base = pb*32*512
                        e, stage = divmod(s-base, n)
                        inter.append(base+stage*32+e)
                    offsets = {'current': ids*STAGE,
                               'interleaved': torch.tensor(inter, dtype=torch.int64, device='cuda')*STAGE}
                    offsets['repeat'] = offsets['current']
                    bounds_gpu = torch.tensor(bounds, dtype=torch.int32, device='cuda')
                    expected = torch.zeros((GRID, 128, 2), dtype=torch.int64, device='cuda')
                    for c in range(GRID):
                        expected[c].copy_(stage_checksums.index_select(0, ids[bounds[c]:bounds[c+1]]).sum(0))
                    outputs = {a: torch.empty_like(expected) for a in ARMS}
                    expected_completion = {}
                    for arm in ARMS:
                        expected_completion[arm] = torch.stack((
                            torch.tensor([len(s)*STAGE for s in streams], dtype=torch.int64, device='cuda'),
                            torch.stack([offsets[arm][bounds[c]:bounds[c+1]].sum() for c in range(GRID)])), 1)
                    def validate_stream(arm):
                        view = views['current' if arm == 'repeat' else arm]
                        ext.run(view, offsets[arm], bounds_gpu, outputs[arm], True)
                        if not torch.equal(outputs[arm], expected):
                            raise RuntimeError('Every-byte stream checksum failed: ' + arm)
                    graphs = {}
                    for arm in ARMS:
                        view = views['current' if arm=='repeat' else arm]
                        validate_stream(arm)
                        fn = lambda a=arm, v=view: ext.run(v, offsets[a], bounds_gpu, outputs[a], False)
                        graphs[arm] = capture(torch, fn)
                        if not torch.equal(outputs[arm][:, 0, :], expected_completion[arm]):
                            raise RuntimeError('Pre-timing transfer completion failed: '+arm)
                    events = {a: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for a in ARMS}
                    for pair in events.values():
                        for event in pair:
                            event.record()
                    torch.cuda.synchronize()
                    for rep in range(args.reps):
                        rotation = (rep+offset_index+args.route_seeds.index(seed))%3
                        order = list(ARMS[rotation:]+ARMS[:rotation])
                        if (rep//3)%2:
                            order.reverse()
                        for _ in range(3):
                            for arm in order:
                                graphs[arm].replay()
                        torch.cuda.synchronize()
                        record = {'rep': rep, 'order': order, 'us': {}}
                        row['records'].append(record)
                        for arm in order:
                            sink = flush.sum()  # read-only 256 MiB; OUTSIDE event interval
                            begin, end = events[arm]
                            begin.record()
                            graphs[arm].replay()
                            end.record()
                            end.synchronize()
                            us = begin.elapsed_time(end)*1000
                            if not math.isfinite(us) or us <= 0:
                                raise RuntimeError('Nonfinite or nonpositive CUDA event time')
                            record['us'][arm] = us
                            del sink
                        record['completed_bytes_and_offsets_equal'] = {
                            a: torch.equal(outputs[a][:, 0, :], expected_completion[a]) for a in ARMS}
                        if not all(record['completed_bytes_and_offsets_equal'].values()):
                            raise RuntimeError('Timed transfer completion mismatch')
                        for arm in ARMS:
                            validate_stream(arm)
                        record['checksums_equal'] = {a: True for a in ARMS}
                        save(output, report)
                    row['median_us'] = {a: statistics.median(r['us'][a] for r in row['records']) for a in ARMS}
                    row['effective_TB_s'] = {a: row['logical_weight_bytes']/t/1e6 for a, t in row['median_us'].items()}
                    row['paired_ratios'] = [r['us']['current']/r['us']['interleaved'] for r in row['records']]
                    row['aa_ratios'] = [r['us']['current']/r['us']['repeat'] for r in row['records']]
                    row['aa_within_one_percent'] = abs(statistics.median(row['aa_ratios'])-1)<=0.01
                    row['conditional_saved_us'] = statistics.median(
                        (r['us']['current']+r['us']['repeat'])/2-r['us']['interleaved'] for r in row['records'])
                    row['cta_expected_checksum_pairs'] = expected.sum(1).cpu().tolist()
                    row['paired_saved_us'] = [
                        (r['us']['current']+r['us']['repeat'])/2-r['us']['interleaved']
                        for r in row['records']]
                    row['status'] = 'passed' if row['aa_within_one_percent'] else 'noise_gate_failed'
                    del graphs, outputs, offsets, ids, bounds_gpu, expected
                    save(output, report)
            after = {k: validate_layout(torch, packed, v, k=='interleaved', stage_checksums) for k, v in views.items()}
            for row in report['cases']:
                if row['offset_bytes']==offset:
                    row['correctness_after'] = after
            save(output, report)
        savings = {name: min(r['conditional_saved_us'] for r in report['cases'] if r['load']==name) for name in LOADS}
        report['conservative_across_offsets_and_seeds_saved_us'] = savings
        report['median_across_offsets_and_seeds_saved_us'] = {
            name: statistics.median(r['conditional_saved_us'] for r in report['cases'] if r['load']==name) for name in LOADS}
        report['optimistic_across_offsets_and_seeds_saved_us'] = {
            name: max(r['conditional_saved_us'] for r in report['cases'] if r['load']==name) for name in LOADS}
        gate = all(r['aa_within_one_percent'] for r in report['cases'])
        report['board'] = {'weights': 'six equal loads; D1/D16/V1 unchanged', 'aa_gate_passed': gate,
                           'status': 'inconclusive_without_measured_full_kernel_denominators',
                           'formula': 'exp(sum(log(t_load/(t_load-max(0,saved_us))))/6)-1',
                           'interpretation': 'conditional port ceiling only; standalone stream gains are never credited as real kernel gains'}
        if args.full_kernel_us:
            times = json.loads(Path(args.full_kernel_us).read_text())
            report['board']['denominators_source_sha256'] = digest(Path(args.full_kernel_us))
            report['board']['baseline_us'] = times
            gains = []
            for name in LOADS:
                t = float(times[name])
                saving = max(0, savings[name])
                if not math.isfinite(t) or not 0 <= saving < t:
                    raise ValueError('Invalid measured denominator or synthetic saving >= whole kernel: '+name)
                gains.append(math.log(t/(t-saving)))
            estimate = math.exp(sum(gains)/6)-1
            report['board'].update(conditional_fraction=estimate,
                                   status=('conditional_ceiling_at_least_3_percent' if estimate>=0.03 else 'conditional_ceiling_below_3_percent') if gate else 'inconclusive_aa_failed')
        report['status'] = 'complete'
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        report['status'] = 'error'
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        save(output, report)
    return 0 if report['status']=='complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weight-seed', type=int, default=731283)
    parser.add_argument('--route-seeds', type=int, nargs='+', default=[731283, 20261031])
    parser.add_argument('--reps', type=int, default=9)
    parser.add_argument('--timeout-seconds', type=int, default=840)
    parser.add_argument('--full-kernel-us')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reps<9 or len(set(args.route_seeds))<2 or not 1<=args.timeout_seconds<=880:
        parser.error('Require >=9 reps, >=2 distinct route seeds, timeout 1..880 seconds')
    args.route_seeds = list(dict.fromkeys(args.route_seeds))
    output = Path(args.output)
    if output.suffix != '.json':
        output = Path(str(output)+'.json')
    if args.worker:
        return worker(args, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, {'status': 'launching', 'errors': [], 'cases': []})
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], start_new_session=True)
    try:
        return proc.wait(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        report = json.loads(output.read_text())
        report['status'] = 'timeout'
        report.setdefault('errors', []).append({'type': 'DeadlineExceeded', 'seconds': args.timeout_seconds})
        save(output, report)
        return 124

if __name__ == '__main__':
    raise SystemExit(main())
