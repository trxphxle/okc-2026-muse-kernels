"""Round4 real-routing collector and unchanged, isolated expert profiling worker.

External diagnostics only; never a submission or a kernel candidate. The collector
runs once per weight seed and freezes the actual full-layer expert boundary. The
profile worker consumes that boundary without regenerating synthetic routing.
The parent runner owns the hard process-group deadline and raw NCU evidence.
"""
from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

from round4_common import (SOURCE_SHA, WEIGHT_SEEDS, FLUSH_BYTES, REL_L2_TOL,
    descriptors, plan, sha, save, load, load_fixture, expert_fn, capture, measure)


def numerical(actual, gold):
    import torch
    a, g = actual.float(), gold.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(g).all())
    if not finite:
        return dict(passed=False, finite=False, tolerance=REL_L2_TOL)
    delta = a - g
    relative = float(delta.norm() / g.norm().clamp_min(1e-12))
    row_norm = g.reshape(-1, g.shape[-1]).norm(dim=1)
    row_floor = (row_norm.mean() * .1).clamp_min(1e-12)
    worst_row = float((delta.reshape(-1, delta.shape[-1]).norm(dim=1)
                       / torch.maximum(row_norm, row_floor)).max())
    return dict(passed=relative <= REL_L2_TOL and worst_row <= 5 * REL_L2_TOL,
                finite=True, relative_l2=relative, worst_row_relative_l2=worst_row,
                tolerance=REL_L2_TOL, row_tolerance=5 * REL_L2_TOL,
                max_abs=float(delta.abs().max()))


def require(check, label):
    if not check['passed']:
        raise RuntimeError(label + ': ' + repr(check))


def weight_hash(bench, state):
    return bench.fingerprint({str(i): t for i, t in enumerate(state['ma_w'])})


def prepare(module, reference, bench, seed):
    import torch
    start = time.monotonic()
    weights = reference.generate_weights(seed)
    original_hash = bench.fingerprint(weights)
    private = bench.clone_data(weights)
    state = module.prepare(private)
    torch.cuda.synchronize()
    if bench.fingerprint(weights) != original_hash:
        raise RuntimeError('Prepare mutated independent reference weights')
    metadata = dict(weight_seed=seed, seconds=time.monotonic() - start,
                    raw_weights_sha256=original_hash,
                    prepared_expert_weights_sha256=weight_hash(bench, state),
                    independent_reference_storage=True,
                    ma_sfx=list(state['ma_sfx']), nsm=state['nsm'])
    return weights, private, state, metadata


def paired_aa(fn, tensors, flush):
    cold_a, cold_b = capture(fn, flush=flush), capture(fn, flush=flush)
    for name, graph in (('A', cold_a), ('A_repeat', cold_b)):
        require(numerical(graph['output'], tensors['gold']), name + ' precheck')
    samples = []
    graphs = {'A': cold_a, 'A_repeat': cold_b}
    for i in range(12):
        order = ('A', 'A_repeat') if i % 2 == 0 else ('A_repeat', 'A')
        times = {name: measure(graphs[name]) for name in order}
        samples.append(dict(pair=i, order=list(order), us=times))
    for graph in (cold_a, cold_b):
        require(numerical(graph['output'], tensors['gold']), 'A/A postcheck')
    ratios = [row['us']['A'] / row['us']['A_repeat'] for row in samples]
    return dict(samples=samples, median_A_us=statistics.median(row['us']['A'] for row in samples),
        median_A_repeat_us=statistics.median(row['us']['A_repeat'] for row in samples),
        median_paired_ratio=statistics.median(ratios),
        within_one_percent=abs(statistics.median(ratios)-1) <= .01,
        interval='Collector process without NCU attached: internal CUDA events around native PDL0 expert only; graph256MiB reduction precedes begin event',
        used_for_dram_bandwidth=False)


def collect_case(module, reference, bench, weights, state, descriptor, preparation, flush):
    import torch
    shape = {key: descriptor[key] for key in
             ('batch', 'q_len', 'ctx', 'layer', 'page', 'spread', 'dist')}
    fresh = reference.generate_input(seed=descriptor['input_seed'], **shape)
    private = bench.clone_data(fresh)
    probe = module.kernel_fn(private, state)
    global_ids = probe[2].long()
    if not bool(((global_ids >= 0) & (global_ids < 256)).all()):
        raise RuntimeError('Full-layer probe returned invalid global expert IDs')
    shard_counts = torch.bincount((global_ids // 32).flatten(), minlength=8)
    local_start = 32 * int(shard_counts.argmax())
    probe_ids = global_ids.clone()
    del probe, global_ids
    fresh['local_start'].fill_(local_start)
    bench.refill(private, fresh)
    original_factory = module._moe
    boundary = {}
    calls = 0

    class Proxy:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def okc_run(self, *args):
            nonlocal calls
            calls += 1
            if calls != 1:
                raise RuntimeError('Expected exactly one native expert call per full layer')
            if len(args) != 9 or args[6:8] != (0, 32):
                raise RuntimeError('Unexpected native expert signature/shard')
            # Clones are diagnostic-only, untimed, and cannot change launch args.
            boundary.update(x=args[2].clone(), ids=args[3].clone(), tw=args[4].clone())
            result = self.native.okc_run(*args)
            boundary['expected'] = args[5].clone()
            return result

    module._moe = lambda *a, **kw: Proxy(original_factory(*a, **kw))
    try:
        output = module.kernel_fn(private, state)
        torch.cuda.synchronize()
    finally:
        module._moe = original_factory
    if calls != 1:
        raise RuntimeError(f'Observed {calls} native calls, expected one')
    ids = output[2].long()
    # The preliminary call only chooses a shard. Freeze the captured call's
    # actual routes; separate executions need not have identical near-boundary
    # top-k decisions. The selected shard must still be busiest for this call.
    route_differences = int((ids != probe_ids).sum())
    actual_shard_counts = torch.bincount((ids // 32).flatten(), minlength=8)
    if int(actual_shard_counts[local_start // 32]) != int(actual_shard_counts.max()):
        raise RuntimeError('Selected shard is not busiest for the actual captured call')
    normalized = (ids - local_start) % 256
    normalized = torch.where(normalized < 32, normalized, -1)
    if not torch.equal(boundary['ids'].long(), normalized):
        raise RuntimeError('Captured local routes disagree with full-layer global output')
    if boundary['x'].shape != (descriptor['m'], 4096):
        raise RuntimeError('Unexpected actual expert boundary shape')
    gold = reference._experts(boundary['x'].float(), boundary['ids'], boundary['tw'], weights)
    boundary['gold'] = gold
    check = numerical(boundary['expected'], gold)
    require(check, 'Full-layer expert boundary versus public _experts')
    layer_check = bench.oracle(private, output, weights, fresh)
    require(layer_check, 'Full-layer public oracle')
    standalone = expert_fn(module, state, boundary)
    standalone_output = standalone()
    torch.cuda.synchronize()
    standalone_check = numerical(standalone_output, gold)
    require(standalone_check, 'Standalone PDL0 versus public _experts')
    boundary_check = numerical(standalone_output, boundary['expected'])
    require(boundary_check, 'Standalone PDL0 versus captured full-layer boundary')
    counts = torch.bincount(normalized[normalized >= 0], minlength=32).cpu().tolist()
    metadata = dict(**descriptor, **plan(counts), local_start=local_start,
        shard_counts=actual_shard_counts.cpu().tolist(), native_call_count=calls,
        probe_shard_counts=shard_counts.cpu().tolist(), probe_capture_route_differences=route_differences,
        preparation=preparation, expert_check=check, full_layer_check=layer_check,
        standalone_check=standalone_check, standalone_boundary_check=boundary_check,
        unprofiled_paired_AA=paired_aa(standalone, boundary, flush),
        input_origin='Actual unchanged qualified full-layer decoder; untimed _moe factory proxy forwarding unchanged okc_run arguments',
        local_id_normalization='(global_id-local_start)%256; nonlocal IDs replaced by -1; weights remain local indices0..31',
        input_without_cache_sha256=bench.fingerprint({k: v for k, v in fresh.items()
                                                     if k not in ('k_cache', 'v_cache')}))
    portable = {key: tensor.detach().cpu() for key, tensor in boundary.items()}
    portable.update(source_sha256=SOURCE_SHA, metadata=metadata,
                    global_ids=output[2].detach().cpu())
    torch.save(portable, descriptor['fixture_file'])
    metadata['fixture_sha256'] = sha(descriptor['fixture_file'])
    return metadata


def frozen_manifest(path):
    import json
    manifest = json.loads(path.read_text())
    if manifest.get('source_sha256') != SOURCE_SHA:
        raise RuntimeError('Manifest source hash mismatch')
    expected = descriptors()
    fixtures = manifest.get('fixtures', [])
    if len(fixtures) != len(expected):
        raise RuntimeError(f'Expected all18 fixtures, found {len(fixtures)}')
    for actual, wanted in zip(fixtures, expected):
        if any(actual.get(k) != v for k, v in wanted.items()):
            raise RuntimeError('Manifest descriptor/order drift: ' + wanted['case_id'])
        if sha(path.parent / actual['fixture_file']) != actual['fixture_sha256']:
            raise RuntimeError('Frozen fixture drift: ' + wanted['case_id'])
    return manifest


def profile_case(module, state, bench, fixture, root, flush, sequence):
    import torch
    path = root / fixture['fixture_file']
    if sha(path) != fixture['fixture_sha256']:
        raise RuntimeError('Fixture changed before target')
    saved, tensors = load_fixture(path)
    if saved['metadata']['case_id'] != fixture['case_id']:
        raise RuntimeError('Fixture identity mismatch')
    counts = torch.bincount(tensors['ids'][tensors['ids'] >= 0].long(), minlength=32).cpu().tolist()
    if plan(counts) != {key: fixture[key] for key in plan(counts)}:
        raise RuntimeError('Frozen route plan drift')
    input_hash = bench.fingerprint(tensors)
    fn = expert_fn(module, state, tensors)
    # This graph contains one native expert launch, no reduction/other kernels.
    target = capture(fn)
    require(numerical(target['output'], tensors['gold']), 'Profile precheck')
    sink = flush.sum()
    torch.cuda.synchronize()
    region = f"round4/{sequence:02d}/{fixture['case_id']}"
    torch.cuda.nvtx.range_push(region)
    torch.cuda.profiler.start()
    try:
        target['graph'].replay()
        target['end'].synchronize()
    finally:
        torch.cuda.profiler.stop()
        torch.cuda.nvtx.range_pop()
    checks = {key: numerical(target['output'], tensors[key]) for key in ('gold', 'expected')}
    for key, check in checks.items():
        require(check, 'Profile target versus ' + key)
    if bench.fingerprint(tensors) != input_hash or sha(path) != fixture['fixture_sha256']:
        raise RuntimeError('Frozen fixture/input mutated during timing')
    del sink
    return dict(case_id=fixture['case_id'], sequence=sequence, nvtx_region=region,
        fixture_sha256=fixture['fixture_sha256'], source_sha256=SOURCE_SHA,
        checks=checks, unprofiled_paired_AA=fixture['unprofiled_paired_AA'], profiler_regions_started=1,
        profiler_regions_completed=1, target_graph_replays=1, pdl=0,
        cold_read_flush_bytes=FLUSH_BYTES,
        profile_interval='One unchanged native expert CUDA kernel in one graph replay between cudaProfilerStart/Stop; PDL0; synchronized256MiB read reduction immediately before start, outside region. NCU gpu__time_duration.sum and dram__bytes_read.sum must share same launch and one metric pass.',
        profiled_event_us=target['begin'].elapsed_time(target['end'])*1000,
        profiled_event_warning='NCU perturbed; never used as native timing or matched counter denominator')


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('collect', 'profile'), required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--bench', type=Path, default=Path('compare_decode_public.py'))
    parser.add_argument('--reference', type=Path, default=Path('reference.py'))
    parser.add_argument('--manifest', type=Path, default=Path('round4-fixtures.json'))
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--resume-report', type=Path)
    args = parser.parse_args()
    report_path = Path(f'{args.output_prefix}-pass-{os.getpid()}-{time.time_ns()}.json')
    started = time.monotonic()
    report = dict(status='starting', phase=args.phase, source_sha256=sha(args.baseline),
        cases=[], preparations=[], artifact_files=[], pid=os.getpid(),
        requested_case_ids=[d['case_id'] for d in descriptors()])
    manifest = dict(source_sha256=SOURCE_SHA, fixtures=[])
    prior_collection_seconds = 0.0

    def checkpoint():
        report['elapsed_seconds'] = time.monotonic() - started
        report['missing_case_ids'] = [x for x in report['requested_case_ids']
                                     if x not in {c['case_id'] for c in report['cases']}]
        save(report_path, report)

    try:
        checkpoint()
        if report['source_sha256'] != SOURCE_SHA:
            raise RuntimeError('Qualified decoder source mismatch')
        import torch
        sys.path.insert(0, str(args.reference.resolve().parent))
        reference = load(args.reference, 'reference')
        bench = load(args.bench, '_round4_bench')
        bench.torch, bench.reference = torch, reference
        module = load(args.baseline, '_round4_control')
        report['source_files'] = {str(p): sha(p) for p in
            (args.baseline, args.bench, args.reference, args.reference.with_name('checks.py'),
             args.reference.with_name('nvfp4.py'), Path(__file__), Path(__file__).with_name('round4_common.py'))}
        report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), python=sys.version)
        if args.phase == 'profile':
            manifest = frozen_manifest(args.manifest)
            report['manifest_sha256'] = sha(args.manifest)
        elif args.resume_report is not None:
            previous = json.loads(args.resume_report.read_text())
            manifest = json.loads(args.manifest.read_text())
            if (previous.get('phase') != 'collect' or previous.get('source_sha256') != SOURCE_SHA
                    or manifest.get('source_sha256') != SOURCE_SHA):
                raise RuntimeError('Cannot resume a different collector/source')
            expected_prefix = descriptors()[:len(manifest['fixtures'])]
            if previous['cases'] != manifest['fixtures']:
                raise RuntimeError('Partial manifest differs from its preserved collector evidence')
            for row, descriptor in zip(manifest['fixtures'], expected_prefix):
                if any(row.get(k) != v for k, v in descriptor.items()):
                    raise RuntimeError('Partial fixture descriptor drift')
                if sha(args.manifest.parent / row['fixture_file']) != row['fixture_sha256']:
                    raise RuntimeError('Partial fixture byte drift')
            for path, digest in previous['source_files'].items():
                if Path(path).name != Path(__file__).name and report['source_files'].get(path) != digest:
                    raise RuntimeError('Resume dependency changed: ' + path)
            prior_collection_seconds = previous.get('prior_collection_seconds', 0.0) + previous['elapsed_seconds']
            report['prior_collection_seconds'] = prior_collection_seconds
            report['resumed_case_ids'] = [row['case_id'] for row in manifest['fixtures']]
            report['cases'] = list(manifest['fixtures'])
            manifest.setdefault('collection_history', []).append(dict(
                report=args.resume_report.name, source_files=previous['source_files'],
                completed_case_ids=report['resumed_case_ids'], status=previous['status']))
        elif args.manifest.exists() or any(Path(d['fixture_file']).exists() for d in descriptors()):
            raise RuntimeError('Refusing to overwrite frozen fixtures; supply the preserved --resume-report')
        flush = torch.ones(FLUSH_BYTES//4, dtype=torch.float32, device='cuda')
        sequence = 0
        for seed in WEIGHT_SEEDS:
            rows = descriptors() if args.phase == 'collect' else manifest['fixtures']
            completed = {row['case_id'] for row in report['cases']}
            selected = [row for row in rows if row['weight_seed'] == seed
                        and (args.phase == 'profile' or row['case_id'] not in completed)]
            if not selected:
                continue
            weights, private_weights, state, preparation = prepare(module, reference, bench, seed)
            report['preparations'].append(preparation)
            checkpoint()
            for descriptor in selected:
                if args.phase == 'collect':
                    row = collect_case(module, reference, bench, weights, state, descriptor, preparation, flush)
                    manifest['fixtures'].append(row)
                    report['artifact_files'].append(row['fixture_file'])
                    save(args.manifest, manifest)
                else:
                    expected = descriptor['preparation']
                    for key in ('raw_weights_sha256', 'prepared_expert_weights_sha256', 'ma_sfx', 'nsm'):
                        if preparation[key] != expected[key]:
                            raise RuntimeError('Source/weights/build drift: ' + key)
                    row = profile_case(module, state, bench, descriptor, args.manifest.parent, flush, sequence)
                report['cases'].append(row)
                sequence += 1
                checkpoint()
            if weight_hash(bench, state) != preparation['prepared_expert_weights_sha256']:
                raise RuntimeError('Prepared expert weights mutated')
            del weights, private_weights, state
            gc.collect()
            torch.cuda.empty_cache()
        if sha(args.baseline) != SOURCE_SHA:
            raise RuntimeError('Source drift during run')
        if args.phase == 'collect':
            manifest['collection_seconds'] = prior_collection_seconds + time.monotonic() - started
            manifest['source_files'] = report['source_files']
            manifest.setdefault('collection_history', []).append(dict(
                report=report_path.name, source_files=report['source_files'],
                completed_case_ids=[row['case_id'] for row in report['cases']
                                    if row['case_id'] not in report.get('resumed_case_ids', [])],
                status='complete'))
            save(args.manifest, manifest)
            report['artifact_files'].append(args.manifest.name)
        else:
            if sha(args.manifest) != report['manifest_sha256']:
                raise RuntimeError('Manifest drift during profile')
        report.update(status='complete', peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc),
                                                traceback=traceback.format_exc()))
    finally:
        checkpoint()
        print(report_path.name, flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
