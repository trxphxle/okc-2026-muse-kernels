"""One unchanged whole-layer graph, one CUDA-profiler region, one case.

Invoked by cold_ncu.py, not a submission. Application replay starts this process
again for each metric pass. Each invocation retains a distinct JSON/PT pair;
no pass overwrites another pass's correctness or fixture evidence. Preparation,
capture, helper warmups, input refill and the 256 MiB read flush are OUTSIDE the
selected profiler range. The flush does not prove that every instruction-cache
level is cold. Warm mode is a combined-cache diagnostic, not an optimization.

Source caveats: COORDINATION.md Round 2; alygg77 H200_MEASURED_FACTS.md sections
2/3 (PDL and allocation effects), NEGATIVE_RESULTS.md decode sections 1-4;
research/FINDINGS.md R1. No donor code imported. The hash-locked raw decoder is
missing mandatory BF16 boundaries: even an oracle pass is not qualification.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

SOURCE_SHA = '4531458701f45eb713fcc885447149a563826316edea1f2a419a25b4f7c96964'
CASES = ('D1-sliding', 'D1-full', 'V1-sliding', 'V1-full')
FLUSH_BYTES = 256 * 1024 * 1024


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--bench', type=Path, default=Path('compare_decode_public.py'))
    parser.add_argument('--reference', type=Path, default=Path('reference.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--cache-mode', choices=('cold', 'warm'), default='cold')
    parser.add_argument('--input-offset', type=int, default=1)
    args = parser.parse_args()
    if args.output.name != str(args.output) or args.output.suffix != '.json':
        parser.error('--output must be a JSON basename for run_layer retrieval')
    prefix = args.output.stem + f'-pass-{os.getpid()}-{time.time_ns()}'
    output_path = Path(prefix + '.json')
    raw_path = Path(prefix + '.pt')
    started = time.monotonic()
    report = dict(status='starting', case_id=args.case, cache_mode=args.cache_mode,
                  pid=os.getpid(), artifact_files=[], official_score=False,
                  submission_allowed=False, source_sha256=sha(args.baseline),
                  weight_seed=731283, input_offset=args.input_offset,
                  profiler_regions_started=0, profiler_regions_completed=0,
                  target_graph_replays=0,
                  precision_warning='Raw baseline lacks mandatory BF16 boundaries; diagnostic only',
                  protocol='One capture via bench.capture(flush=None), helper warmups only; refill then one whole-layer target replay; no per-child flush',
                  flush_bytes=FLUSH_BYTES if args.cache_mode == 'cold' else 0)
    save(output_path, report)
    try:
        if report['source_sha256'] != SOURCE_SHA:
            raise RuntimeError('Baseline SHA mismatch; refusing changed decoder')
        import torch
        # Keep reference/checks/nvfp4 imports local to the uploaded harness.
        sys.path.insert(0, str(args.reference.resolve().parent))
        reference = load(args.reference, 'reference')
        bench = load(args.bench, '_cold_ncu_bench')
        bench.torch, bench.reference = torch, reference
        report['sources'] = {role: dict(path=str(path), sha256=sha(path)) for role, path in {
            'baseline': args.baseline, 'bench': args.bench, 'reference': args.reference,
            'checks': args.reference.with_name('checks.py'),
            'nvfp4': args.reference.with_name('nvfp4.py'), 'worker': Path(__file__),
        }.items()}
        report['environment'] = dict(python=sys.version, platform=platform.platform(),
            torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
            env={k: v for k, v in os.environ.items() if k.startswith(
                ('CUDA', 'TORCH', 'TRITON', 'OKC', 'MOE', 'BFP', 'DL_', 'MAX_JOBS'))})
        save(output_path, report)
        module = bench.load(args.baseline, 'cold_ncu_baseline')
        before = time.monotonic()
        weights = reference.generate_weights(report['weight_seed'])
        torch.cuda.synchronize()
        report['weight_generation_seconds'] = time.monotonic() - before
        # prepare retains raw .contiguous() aliases; never share those with oracle weights.
        private_weights = bench.clone_data(weights)
        torch.cuda.synchronize()
        before = time.monotonic()
        report['status'] = 'preparing'
        save(output_path, report)
        state = module.prepare(private_weights)
        torch.cuda.synchronize()
        report['prepare'] = dict(seconds=time.monotonic() - before,
            independent_weight_storage=True,
            router_bias_unchanged=bool(torch.equal(private_weights['router_bias'], weights['router_bias'])),
            mode=bench.mode_state(module, state))
        save(output_path, report)
        if report['prepare']['seconds'] > 300:
            raise RuntimeError('Unchanged baseline prepare exceeded 300 seconds')
        load_name, layer = args.case.split('-')
        q_len = 1 if load_name == 'D1' else 8
        shape = dict(batch=1, q_len=q_len, ctx=80000, layer=layer, page=1,
                     spread='fixed', dist='soft')
        seed_base = 919233 + 100 + q_len * 1000 + (layer == 'full') * 10000
        seed = seed_base + args.input_offset
        fresh = reference.generate_input(seed=seed, **shape)
        private = bench.clone_data(fresh)
        # Probe private caches only, before capture; keep pristine oracle input intact.
        probe = module.kernel_fn(private, state)
        ids = probe[2].long()
        if not bool(((ids >= 0) & (ids < 256)).all()):
            raise RuntimeError('Baseline eager routing produced invalid expert IDs')
        counts = torch.bincount((ids // 32).flatten(), minlength=8)
        local_start = 32 * int(counts.argmax())
        fresh['local_start'].fill_(local_start)
        report.update(shape=shape, seed_base=seed_base, input_seed=seed,
            local_start=local_start, baseline_probe_shard_counts=counts.cpu().tolist(),
            input_sha256=bench.fingerprint(fresh), input_metadata=bench.metadata(fresh))
        del probe, ids, counts
        bench.refill(private, fresh)
        flush = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device='cuda')
        sink = flush.sum()  # initialize the same reduction outside the selected range
        torch.cuda.synchronize()
        report['status'] = 'capturing'
        save(output_path, report)
        captured = bench.capture(module.kernel_fn, private, state, flush=None)
        # The final source override binds _GLE_W_CUR from this sole state during
        # capture. No source globals or launch attributes are patched here.
        report['capture'] = dict(count=1, eager_warmups=3, graph_warmups=3,
            includes_flush=False, external_events=True,
            gle_weight_bound_to_state=getattr(module, '_GLE_W_CUR', None) is state.get('g_preffn'))
        bench.refill(private, fresh)
        torch.cuda.synchronize()
        report.update(status='target_ready', nvtx_region=f'cold_ncu/{args.case}/{args.cache_mode}')
        save(output_path, report)
        if args.cache_mode == 'cold':
            sink = flush.sum()
            torch.cuda.synchronize()
        else:
            # Refill may displace cache lines, so replay once after refill, then
            # immediately target the same graph. This warms code AND data.
            captured['graph'].replay()
            captured['end'].synchronize()
        torch.cuda.nvtx.range_push(report['nvtx_region'])
        torch.cuda.profiler.start()
        report['profiler_regions_started'] = 1
        try:
            captured['graph'].replay()
            report['target_graph_replays'] = 1
            captured['end'].synchronize()
        finally:
            torch.cuda.profiler.stop()
            torch.cuda.nvtx.range_pop()
        report['profiler_regions_completed'] = 1
        event_us = captured['begin'].elapsed_time(captured['end']) * 1000
        if not math.isfinite(event_us) or event_us <= 0:
            raise RuntimeError('Invalid graph event duration under profiler')
        report['profiled_graph_event_us'] = event_us
        report['event_warning'] = 'NCU-injected/possibly serialized; not native timing or a speedup'
        report['routes'] = bench.routes(captured['output'], local_start)
        raw = dict(output=[tensor.detach().cpu() for tensor in captured['output']],
            new_k=private['k_cache'].view(-1, reference.HKV, reference.HEAD_DIM)[reference.new_slots(private)].cpu(),
            new_v=private['v_cache'].view(-1, reference.HKV, reference.HEAD_DIM)[reference.new_slots(private)].cpu(),
            input_without_caches={key: value.cpu() if isinstance(value, torch.Tensor) else value
                                  for key, value in fresh.items() if key not in ('k_cache', 'v_cache')},
            shape=shape, input_seed=seed, weight_seed=report['weight_seed'],
            input_sha256=report['input_sha256'], source_sha256=SOURCE_SHA)
        torch.save(raw, raw_path)
        report['artifact_files'].append(raw_path.name)
        report['status'] = 'checking_oracle'
        save(output_path, report)
        report['oracle'] = bench.oracle(private, captured['output'], weights, fresh)
        # Keep counter evidence even if the diagnostic-only raw base fails its
        # public oracle; never turn that into a correctness or submission claim.
        report['numerically_valid'] = report['oracle']['passed']
        report['status'] = 'complete'
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        del sink
    except BaseException as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc),
                                                 traceback=traceback.format_exc()))
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        save(output_path, report)
        print(json.dumps(dict(worker_result=output_path.name, status=report['status'])), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
