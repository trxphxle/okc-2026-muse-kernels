"""One diagnostic run: Hopper compression and expert VMM backing, not a candidate.

Upload vmm_probe.py, vmm_memory.cu, vmm_dense.py, vmm_experts.py,
decode_cold_control.py, decode reference.py/checks.py/nvfp4.py and moe_reference.py.
All allocation/repacking/validation is outside timed CUDA graphs. Each graph
contains a 256MiB read flush and internal external events around the stage only.
No stage math, weights, precision, or kernel scheduling is changed by this file.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
from torch.utils.cpp_extension import load as load_extension

BASE_SHA = '4531458701f45eb713fcc885447149a563826316edea1f2a419a25b4f7c96964'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def time_arms(arms, flush, base, repeat, validate, reps=36):
    """Fixed fresh fixture, balanced cold graphs; validate actual captured outputs."""
    names = tuple(arms)
    if reps % len(names):
        raise ValueError('Repetitions must balance every arm position')
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for fn in arms.values():
            for _ in range(3):
                fn()
    current.wait_stream(stream)
    torch.cuda.synchronize()
    graphs, outputs, events, sinks = {}, {}, {}, {}
    for name, fn in arms.items():
        begin = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        begin.record(); end.record()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(current)
        with torch.cuda.graph(graph, stream=stream):
            sink = flush.sum()
            begin.record()
            result = fn()
            end.record()
        current.wait_stream(stream)
        graphs[name], outputs[name], events[name], sinks[name] = graph, result, (begin, end), sink
        for _ in range(3):
            graph.replay()
        end.synchronize()
    checks_before = validate(outputs)
    records = []
    for rep in range(reps):
        order = names[rep % len(names):] + names[:rep % len(names)]
        if (rep // len(names)) % 2:
            order = tuple(reversed(order))
        # Avoid an idle-clock first-arm penalty after host-side validation.
        for _ in range(2):
            for name in order:
                graphs[name].replay()
        torch.cuda.synchronize()
        times = {}
        for name in order:
            graphs[name].replay()
            begin, end = events[name]
            end.synchronize()
            us = begin.elapsed_time(end) * 1000
            if not math.isfinite(us) or us <= 0:
                raise RuntimeError('Invalid internal graph event duration')
            times[name] = us
        records.append({'rep': rep, 'order': order, 'us': times})
    checks_after = validate(outputs)
    ratios = {name: statistics.median(r['us'][base] / r['us'][name] for r in records)
              for name in names}
    result = dict(median_us={name: statistics.median(r['us'][name] for r in records) for name in names},
                  median_paired_ratio=ratios, aa_ratio=ratios[repeat],
                  aa_within_one_percent=abs(ratios[repeat] - 1) <= .01,
                  records=records, checks_before=checks_before, checks_after=checks_after,
                  base=base, repeat=repeat)
    torch.cuda.synchronize()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='decode_cold_control.py')
    parser.add_argument('--output', default='vmm-probe.json')
    args = parser.parse_args()
    output = Path(args.output)
    report = dict(status='running', diagnostic_only=True, weight_seed=731283,
                  stage_gain_gate=1.03, errors=[], dense={}, experts={}, artifact_files=[])
    started = time.monotonic()

    def checkpoint():
        report['elapsed_seconds'] = time.monotonic() - started
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(output)

    try:
        paths = [Path(args.baseline), Path(__file__), Path('vmm_memory.cu'),
                 Path('vmm_dense.py'), Path('vmm_experts.py'), Path('reference.py'),
                 Path('moe_reference.py'), Path('checks.py'), Path('nvfp4.py')]
        report['sources'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        if report['sources'][Path(args.baseline).name] != BASE_SHA:
            raise RuntimeError('Changed baseline: expected unchanged lane copy')
        report['environment'] = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                      cuda=torch.version.cuda)
        checkpoint()
        ext = load_extension(name='vmm_probe_' + report['sources']['vmm_memory.cu'][:12],
            sources=['/workspace/vmm_memory.cu'], extra_cuda_cflags=['-O3', '-std=c++20',
            '-gencode=arch=compute_90a,code=sm_90a'],
            extra_ldflags=['-L/usr/local/cuda/lib64/stubs', '-lcuda'], verbose=False)
        report['capabilities'] = ext.capabilities()
        checkpoint()
        reference = load_module('reference.py', 'reference')
        moe_reference = load_module('moe_reference.py', 'vmm_moe_reference')
        module = load_module(args.baseline, 'vmm_control')
        weights = reference.generate_weights(731283)
        before = time.monotonic()
        state = module.prepare(weights)
        torch.cuda.synchronize()
        report['baseline_prepare_seconds'] = time.monotonic() - before
        report['precision_warning'] = 'Raw upstream arithmetic retained for backing comparison; numerical pass is not submission qualification.'
        checkpoint()
        flush = torch.ones(256 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)
        from vmm_dense import run_dense
        from vmm_experts import run_experts
        run_dense(module, weights, state, ext, flush, time_arms, exact, report, checkpoint)
        gc.collect(); torch.cuda.synchronize()
        run_experts(module, weights, state, ext, moe_reference, flush, time_arms, exact, report, checkpoint)
        gc.collect(); torch.cuda.synchronize()
        report['vmm_lifetime'] = ext.live_stats()
        if any(report['vmm_lifetime'][key] for key in ('allocations', 'bytes', 'cleanup_errors')):
            raise RuntimeError('VMM allocations or cleanup errors remain after all captures leave scope')
        report['peak_torch_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['status'] = 'complete'
    except BaseException as exc:
        report['status'] = 'failed'
        report['errors'].append(dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        checkpoint()
        print(json.dumps({'status': report['status'], 'output': str(output),
                          'capabilities': report.get('capabilities'), 'errors': report['errors']}), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
