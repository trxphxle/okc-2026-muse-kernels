"""Diagnostic only: cold versus warmed decode-layer replay, NEVER a candidate.

CLI (upload beside compare_decode_public.py/reference.py/checks.py/nvfp4.py):
  python cold_ceiling.py --baseline decode_cold_control.py --output ceiling.json

Four cases only: D1/V1 x sliding/full; three fresh seeds; --reps balanced
repetitions per arm per seed (default twelve). Internal external-event timing:
256 MiB read-only sweep -> optional replay -> begin -> target -> end, all in
one CUDA graph. No host gap, side stream fork, removed launch, or changed PDL.
Independent module instances and cloned weights precede separate prepare calls.
The unchanged prepare itself can tune larger shapes; this diagnostic does not
benchmark the other four loads, which are priced at ratio 1 in board estimates.

Native CUPTI profiles use a profiler-only built-in CUDA Sleep delimiter between
preconditioning and begin. Parsing selects only native kernels AFTER that GPU
delimiter completes, not CPU NVTX timestamps or a guessed suffix length. Raw and
filtered Chrome traces are retained. Delimiter overhead/cache perturbation is
NOT in unprofiled timings; profiled kernel deltas are attribution evidence, never
summed into critical-path gain. All target kernel dependencies remain intact.

A sweep is not proof that all instruction caches are cold. Whole-layer shadow
warming cannot guarantee late merge code survives earlier kernels. Same-data
warmth is a combined code/data-cache upper experiment, not an optimization or a
universal bound. Shared opaque resources, independently tuned plans, TLB/cache
placement, and hidden native scratch preclude claiming isolated instruction
warming. Existing missing BF16 boundaries in the raw base remain a separate
submission blocker even if public numerical checks pass. No donor code imported.

Source rationale: COORDINATION.md Round 2; alygg77 NEGATIVE_RESULTS sections 1.1-
1.4 and H200_MEASURED_FACTS sections 2-3; research/FINDINGS.md R1. The donor's
10.8->8.9 us is own-base evidence only. Parent owns NCU residual no_instruction
analysis and any subsequent safe intra-kernel warming experiment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import types

BASE_SHA = '4531458701f45eb713fcc885447149a563826316edea1f2a419a25b4f7c96964'
ARMS = ('cold', 'cold_repeat', 'same_data', 'shadow')
WEIGHT = {'sliding': 46, 'full': 16}
FLUSH_BYTES = 256 * 1024 * 1024


def save(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def initial(args):
    return dict(status='running', diagnostic_only=True, submission_allowed=False,
        baseline_required_sha256=BASE_SHA, weight_seed=731283, input_offset=args.input_offset,
        fresh_seeds_per_case=3, repetitions_per_arm_per_seed=args.reps, traces_per_arm=2,
        artifact_files=[], errors=[], results=[],
        precision_status='Raw base missing BF16 boundaries: diagnostic only regardless of oracle pass',
        caveats=[
            'Cold means after 256MiB read-only sweep, not proven instruction-cache eviction.',
            'Same-data is combined code/data-cache warm experiment, not legal cross-call reuse.',
            'Same-data prelude re-executes kernel_fn with identical input/state storage; capture-time temporary allocations may differ, so not all target scratch is guaranteed warm.',
            'Shadow is a code-warm/target-data-cold approximation, never a universal instruction bound.',
            'Earlier kernels can evict late merge instructions warmed by a whole-layer shadow.',
            'Opaque native resources and microarchitectural cache/TLB sharing are not disprovable by Python storage audit.',
            'Profiler-only Sleep delimiter may perturb code/cache state; unprofiled events set board estimates.',
            'Native paired kernel duration differences are not additive critical-path savings.',
            'Follow-up intra-kernel ceiling requires parent NCU residual no_instruction evidence.'])


class Prelude:
    """Use capture's flush hook to put all preconditions before internal begin."""
    def __init__(self, flush, module=None, data=None, state=None, marker=False):
        self.flush, self.module = flush, module
        self.data, self.state, self.marker = data, state, marker
        self.output = None

    def sum(self):
        sink = self.flush.sum()
        if self.module is not None:
            self.output = self.module.kernel_fn(self.data, self.state)
        if self.marker:
            torch.cuda._sleep(1)
        return sink


def module_owned(module):
    # Do not descend into imported torch/triton modules or function globals.
    return {key: value for key, value in vars(module).items()
            if not key.startswith('__') and (isinstance(value, (dict, list, tuple, torch.Tensor))
                or type(value).__module__ == module.__name__
                or (isinstance(value, types.ModuleType)
                    and str(getattr(value, '__file__', '')).endswith(('.so', '.pyd'))))}


def inventory(root):
    tensors, opaque, seen = [], [], {}

    def visit(value, path):
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        ident = id(value)
        if ident in seen:
            return
        seen[ident] = path
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            tensors.append(dict(path=path, device=str(value.device), ptr=storage.data_ptr(),
                bytes=storage.nbytes(), shape=list(value.shape), dtype=str(value.dtype),
                storage_offset=value.storage_offset(), stride=list(value.stride())))
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + '[' + repr(key) + ']')
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, path + '[' + str(index) + ']')
        else:
            opaque.append(dict(path=path, identity=ident,
                type=type(value).__module__ + '.' + type(value).__qualname__,
                file=getattr(value, '__file__', None)))
            if not isinstance(value, (types.ModuleType, types.FunctionType, type)):
                attributes = getattr(value, '__dict__', None)
                if isinstance(attributes, dict):
                    visit(attributes, path + '.__dict__')
    visit(root, 'root')
    return dict(tensors=tensors, opaque=opaque)


def alias_proof(target, shadow):
    left, right = inventory(target), inventory(shadow)
    overlaps = []
    # Deduplicate whole storage ranges: views are deliberately conservative.
    def ranges(side):
        unique = {}
        for item in side['tensors']:
            if item['bytes']:
                unique.setdefault((item['device'], item['ptr'], item['bytes']), item['path'])
        return sorted((dev, ptr, ptr + size, path) for (dev, ptr, size), path in unique.items())
    a, b = ranges(left), ranges(right)
    j = 0
    for dev, lo, hi, path in a:
        while j < len(b) and (b[j][0] < dev or (b[j][0] == dev and b[j][2] <= lo)):
            j += 1
        k = j
        while k < len(b) and b[k][0] == dev and b[k][1] < hi:
            if b[k][2] > lo:
                overlaps.append(dict(target=path, shadow=b[k][3], device=dev,
                    overlap_bytes=min(hi, b[k][2]) - max(lo, b[k][1])))
            k += 1
    opaque_ids = {item['identity']: item for item in right['opaque']}
    shared = [dict(target=item, shadow=opaque_ids[item['identity']])
              for item in left['opaque'] if item['identity'] in opaque_ids]
    return dict(target=left, shadow=right, overlapping_storages=overlaps,
        visible_tensor_storage_disjoint=not overlaps, shared_opaque_objects=shared,
        complete_native_isolation_proven=False,
        limitation='Recursive Python containers and object attributes only; native/CUDA hidden buffers are opaque.')


def dispatch(module, state, tokens):
    keys = ('mm_pick', 'moe_pick', 'compose_graft_enabled', 'gluev2', 'marlin_cfg',
            'att_ok', 'ma_ok', 'ma_sfx', 'pbmax', 'nsm', 'compose_choice')
    constants = ('GLE_T', 'GLE_RC', 'GLE_B1X', 'SL_TILE_CONFIG', 'SL_CLUSTER_CONFIG',
                 '_TUNED7', 'OKC_PDL', 'PIN_PICK')
    return dict(tokens=tokens, state={key: bench.metadata(state.get(key)) for key in keys},
        constants={key: bench.metadata(getattr(module, key, None)) for key in constants})


def snapshot(data, output):
    slots = reference.new_slots(data)
    return dict(output=[value.detach().clone() for value in output],
        new_k=data['k_cache'].view(-1, reference.HKV, reference.HEAD_DIM)[slots].clone(),
        new_v=data['v_cache'].view(-1, reference.HKV, reference.HEAD_DIM)[slots].clone())


def bytes_equal(a, b):
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def untouched_cache(data, pristine):
    # Bounded comparisons; do not allocate another full cache or a full-size mask.
    slots = set(reference.new_slots(data).flatten().cpu().tolist())
    result = {}
    for name in ('k_cache', 'v_cache'):
        actual = data[name].view(-1, reference.HKV * reference.HEAD_DIM).view(torch.uint8)
        expected = pristine[name].view_as(data[name]).view(-1, reference.HKV * reference.HEAD_DIM).view(torch.uint8)
        passed = True
        for start in range(0, actual.shape[0], 8192):
            stop = min(start + 8192, actual.shape[0])
            excluded = [slot - start for slot in slots if start <= slot < stop]
            if excluded:
                keep = torch.ones(stop - start, dtype=torch.bool, device=actual.device)
                keep[excluded] = False
                equal = torch.equal(actual[start:stop][keep], expected[start:stop][keep])
            else:
                equal = torch.equal(actual[start:stop], expected[start:stop])
            passed = passed and bool(equal)
        result[name] = passed
    return result


def kernel_signature(item):
    args = item.get('args', {})
    return dict(name=item['name'], geometry={key: args[key] for key in
        ('grid', 'block', 'shared memory', 'registers per thread') if key in args})


def native_trace(captured, path, filtered):
    # Capture/warmup happens before the profiler. Exactly one integrated replay.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        captured['graph'].replay()
        captured['end'].synchronize()
    profiler.export_chrome_trace(str(path))
    trace = json.loads(Path(path).read_text())
    native = sorted((event for event in trace.get('traceEvents', [])
                     if event.get('cat') == 'kernel' and event.get('ph') == 'X'),
                    key=lambda event: (event['ts'], event.get('tid', 0)))
    markers = [event for event in native if any(token in event.get('name', '').lower()
                                               for token in ('sleep', 'spin_kernel'))]
    if len(markers) != 1:
        raise RuntimeError('Expected one native Sleep delimiter, found ' + str(len(markers)))
    marker = markers[0]
    boundary = marker['ts'] + marker['dur']
    crossing = [event for event in native if event is not marker and
                event['ts'] < boundary < event['ts'] + event['dur']]
    if crossing:
        raise RuntimeError('GPU precondition crosses delimiter; cannot isolate target kernels')
    selected = [event for event in native if event is not marker and event['ts'] >= boundary]
    if not selected:
        raise RuntimeError('CUPTI emitted no target native kernels after GPU delimiter')
    # GPU non-kernel events use the same device clock. CPU events are never used.
    extras = [event for event in trace.get('traceEvents', [])
              if str(event.get('cat', '')).startswith(('gpu_memcpy', 'gpu_memset'))
              and event.get('ts', -math.inf) >= boundary]
    save(filtered, dict(traceEvents=selected + extras))
    summary = bench.summarize_trace(filtered)
    summary.update(target_kernel_count=len(selected), excluded_kernel_count=len(native) - len(selected),
        precondition_kernels=[event for event in native if event is not marker and event['ts'] < boundary],
        gpu_delimiter=marker, target_event_us=captured['begin'].elapsed_time(captured['end']) * 1000,
        attribution_method='GPU Sleep completion boundary in one integrated graph; target suffix only',
        profiler_perturbation='One built-in Sleep(1) before begin, absent from timing graphs')
    return summary


def pair_native(cold, warm):
    a, b = cold['launches'], warm['launches']
    matches = len(a) == len(b) and all(kernel_signature(x) == kernel_signature(y) for x, y in zip(a, b))
    return dict(symbol_and_geometry_order_equal=matches,
        cold_kernel_count=len(a), warm_kernel_count=len(b),
        paired=[dict(index=index, name=x['name'], cold_us=x['duration_us'], warm_us=y['duration_us'],
            first_touch_difference_us=x['duration_us'] - y['duration_us'])
            for index, (x, y) in enumerate(zip(a, b))] if matches else [],
        dispatch_mismatches=[dict(index=index, cold=kernel_signature(x), warm=kernel_signature(y))
            for index, (x, y) in enumerate(zip(a, b)) if kernel_signature(x) != kernel_signature(y)],
        warning='Duration differences are NOT summed; overlapping PDL kernels retain native timelines.')


def aggregate(report):
    rows = report['results']
    result = dict(other_load_ratios={name: 1.0 for name in ('D16', 'D32', 'V16', 'V32')},
        per_load={}, complete=False, build_gate=False,
        interpretation='Conditional cache-warming diagnostic only, not a universal bound or candidate forecast')
    for name in ('D1', 'V1'):
        own = [row for row in rows if row['load'] == name and 'median_us' in row]
        if len(own) != 2:
            continue
        means = {arm: sum(WEIGHT[row['layer']] * row['median_us'][arm] for row in own) / 62
                 for arm in ARMS}
        result['per_load'][name] = dict(weighted_us=means,
            ratios={arm: means['cold'] / means[arm] for arm in ARMS},
            saved_us={arm: means['cold'] - means[arm] for arm in ARMS})
    if len(result['per_load']) == 2:
        result['complete'] = True
        result['board_ratios'] = {arm: math.exp(sum(math.log(entry['ratios'][arm])
            for entry in result['per_load'].values()) / 6) for arm in ARMS}
        result['aa_within_one_percent'] = all(row.get('aa_within_one_percent', False) for row in rows)
        result['relative_checks_pass'] = all(record.get('relative_checks_pass', False)
            for row in rows for record in row['records'])
        result['three_percent_ceiling_observed'] = {arm: result['board_ratios'][arm] >= 1.03
                                                   for arm in ('same_data', 'shadow')}
        result['build_gate'] = bool(result['aa_within_one_percent'] and result['relative_checks_pass']
            and any(result['three_percent_ceiling_observed'].values())
            and len(rows) == 4 and all(row.get('status') == 'complete' for row in rows))
        result['gate_caveat'] = 'Parent decision only; no instruction isolation, precision legality, or deployable gain implied'
    report['board_estimate'] = result


def run_case(args, report, modules, states, weights, flush, name, q_len, layer):
    shape = dict(batch=1, q_len=q_len, ctx=80000, layer=layer, page=1, spread='fixed', dist='soft')
    seed = 919233 + 100 + q_len * 1000 + (layer == 'full') * 10000 + args.input_offset
    row = dict(load=name, layer=layer, shape=shape, capture_seed=seed, status='running', records=[], native={})
    report['results'].append(row)
    save(args.output, report)
    fresh = reference.generate_input(seed=seed, **shape)
    data = {role: bench.clone_data(fresh) for role in ('target', 'shadow')}
    captured = {}

    def capture_arm(arm, marker=False):
        role = 'target' if arm == 'same_data' else 'shadow'
        warm = arm in ('same_data', 'shadow')
        prelude = Prelude(flush, modules[role] if warm else None,
                          data[role] if warm else None, states[role] if warm else None, marker)
        graph = bench.capture(modules['target'].kernel_fn, data['target'], states['target'], prelude)
        graph['precondition_output'] = prelude.output
        return graph

    for arm in ARMS:
        captured[arm] = capture_arm(arm)
    row['selected_dispatch'] = {role: dispatch(modules[role], states[role], q_len) for role in modules}
    row['selected_dispatch_equal'] = row['selected_dispatch']['target'] == row['selected_dispatch']['shadow']
    row['state_alias_proof'] = alias_proof(
        dict(state=states['target'], input=data['target'], module_owned=module_owned(modules['target']),
             outputs=[entry['output'] for entry in captured.values()],
             precondition_output=captured['same_data']['precondition_output']),
        dict(state=states['shadow'], input=data['shadow'], module_owned=module_owned(modules['shadow']),
             output=captured['shadow']['precondition_output']))
    binaries = {}
    for role in ('target', 'shadow'):
        binaries[role] = {}
        for item in row['state_alias_proof'][role]['opaque']:
            filename = item.get('file')
            if filename and Path(filename).is_file() and str(filename).endswith(('.so', '.pyd')):
                binaries[role][filename] = sha(filename)
    row['visible_native_binaries'] = binaries
    row['visible_native_binary_files_equal'] = bool(binaries['target']) and binaries['target'] == binaries['shadow']
    row['binary_proof_caveat'] = 'Visible extension files only; lazy/native/library dispatch is not exhaustively inventoried'
    row['shadow_classification'] = 'mixed-cache approximation; isolated instruction warming NOT established'
    row['isolation_reasons'] = [
        'Hidden native buffers / imported opaque resources cannot be proven disjoint',
        'Whole-layer replay does not guarantee late instruction residency or target-data coldness']
    if not row['selected_dispatch_equal']:
        row['isolation_reasons'].append('Independent prepare selected differing dispatch metadata')
    if not row['state_alias_proof']['visible_tensor_storage_disjoint']:
        row['isolation_reasons'].append('Visible target/shadow tensor storage overlaps; see exact paths')
    save(args.output, report)
    prefix = str(Path(args.output).with_suffix('')) + '-' + name + '-' + layer
    for fixture in range(3):
        del fresh
        fresh = reference.generate_input(seed=seed + fixture + 1, **shape)
        bench.refill(data['target'], fresh)
        probe = modules['target'].kernel_fn(data['target'], states['target'])
        counts = torch.bincount((probe[2].long() // 32).flatten(), minlength=8)
        local_start = 32 * int(counts.argmax())
        fresh['local_start'].fill_(local_start)
        del probe
        record = dict(seed=seed + fixture + 1, input_offset=args.input_offset, fixture=fixture,
            local_start=local_start, baseline_probe_shard_counts=counts.cpu().tolist(),
            timings=[], checks={}, routes={})
        row['records'].append(record)
        for role in data:
            bench.refill(data[role], fresh)
        torch.cuda.synchronize()
        # Complete four-position Latin cycles; reverse alternating cycles.
        # Each arm occupies every position exactly args.reps / 4 times.
        for rep in range(args.reps):
            order = ARMS[rep % 4:] + ARMS[:rep % 4]
            if (rep // 4 + fixture) % 2:
                order = tuple(reversed(order))
            # Equal clock conditioning before every group prevents host JSON
            # work from penalizing whichever arm runs first. Each measured
            # graph still applies its own cold-L2/prelude protocol afterward.
            for _ in range(2):
                for arm in order:
                    captured[arm]['graph'].replay()
            torch.cuda.synchronize()
            for arm in order:
                record['timings'].append(dict(rep=rep, arm=arm, us=bench.measure(captured[arm])))
            save(args.output, report)
        record['median_us'] = {arm: statistics.median(item['us'] for item in record['timings']
                                                     if item['arm'] == arm) for arm in ARMS}
        record['aa_ratio'] = record['median_us']['cold'] / record['median_us']['cold_repeat']
        raw = {}
        baseline = None
        # Correctness starts from fresh caches separately for EVERY arm. Repeated
        # graph writes must be idempotent; same-data prelude may not change meaning.
        for arm in ARMS:
            for role in data:
                bench.refill(data[role], fresh)
            bench.measure(captured[arm])
            output = captured[arm]['output']
            current = snapshot(data['target'], output)
            check = dict(oracle=bench.oracle(data['target'], output, weights, fresh),
                         untouched_cache=untouched_cache(data['target'], fresh))
            if baseline is None:
                baseline = current
            check['baseline_relative'] = bench.comparison(current['output'], baseline['output'])
            check['baseline_relative']['cache_equal'] = all(bytes_equal(current[key], baseline[key])
                                                            for key in ('new_k', 'new_v'))
            check['baseline_relative']['output_bitwise_equal'] = all(bytes_equal(x, y)
                for x, y in zip(current['output'], baseline['output']))
            record['checks'][arm] = check
            record['routes'][arm] = bench.routes(output, local_start)
            raw[arm] = {key: [v.cpu() for v in value] if key == 'output' else value.cpu()
                        for key, value in current.items()}
            save(args.output, report)
        record['relative_checks_pass'] = all(check['baseline_relative']['close'] and
            check['baseline_relative']['cache_equal'] and all(check['untouched_cache'].values())
            for check in record['checks'].values())
        record['oracle_all_pass'] = all(check['oracle']['passed'] for check in record['checks'].values())
        record['raw_base_oracle_failed'] = not record['checks']['cold']['oracle']['passed']
        path = prefix + '-fixture' + str(fixture) + '-raw.pt'
        torch.save(dict(seed=record['seed'], shape=shape, local_start=local_start, arms=raw,
            weight_seed=731283, regeneration='generate_input(seed, **shape), then set local_start',
            input_without_caches={key: value.cpu() if isinstance(value, torch.Tensor) else value
                                  for key, value in fresh.items() if key not in ('k_cache', 'v_cache')}), path)
        report['artifact_files'].append(Path(path).name)
        del baseline, current, raw
        save(args.output, report)
    row['median_us'] = {arm: statistics.median(record['median_us'][arm] for record in row['records'])
                        for arm in ARMS}
    row['aa_ratio'] = row['median_us']['cold'] / row['median_us']['cold_repeat']
    row['aa_within_one_percent'] = abs(row['aa_ratio'] - 1) <= .01 and all(
        abs(record['aa_ratio'] - 1) <= .01 for record in row['records'])
    row['native_fixture_seed'] = row['records'][-1]['seed']
    # Fixed fixture, two full integrated replays per condition; no warmups parsed.
    for arm in ARMS:
        row['native'][arm] = []
        profiled = capture_arm(arm, marker=True)
        for repeat in range(2):
            for role in data:
                bench.refill(data[role], fresh)
            torch.cuda.synchronize()
            path = prefix + '-' + arm + '-' + str(repeat) + '-native.json'
            filtered = prefix + '-' + arm + '-' + str(repeat) + '-target.json'
            try:
                summary = native_trace(profiled, path, filtered)
                summary.update(status='complete', trace=Path(path).name, filtered=Path(filtered).name)
            except Exception as exc:
                summary = dict(status='failed', error=repr(exc), traceback=traceback.format_exc())
                torch.cuda.synchronize()
            row['native'][arm].append(summary)
            for artifact in (path, filtered):
                if Path(artifact).is_file():
                    report['artifact_files'].append(Path(artifact).name)
            save(args.output, report)
        del profiled
    row['native_pairs'] = {}
    for arm in ('cold_repeat', 'same_data', 'shadow'):
        row['native_pairs'][arm] = [pair_native(cold, warm) if
            cold['status'] == warm['status'] == 'complete' else dict(status='unavailable')
            for cold, warm in zip(row['native']['cold'], row['native'][arm])]
    row['shadow_native_dispatch'] = []
    for cold, warm in zip(row['native']['cold'], row['native']['shadow']):
        if cold['status'] != 'complete' or warm['status'] != 'complete':
            row['shadow_native_dispatch'].append(dict(status='unavailable'))
            continue
        flush_signature = [kernel_signature(item) for item in cold['precondition_kernels']]
        precondition = [kernel_signature(item) for item in warm['precondition_kernels']]
        target_signature = [kernel_signature(item) for item in warm['launches']]
        flush_matches = precondition[:len(flush_signature)] == flush_signature
        shadow_signature = precondition[len(flush_signature):] if flush_matches else []
        matches = flush_matches and shadow_signature == target_signature
        row['shadow_native_dispatch'].append(dict(status='complete', flush_prefix_matches=flush_matches,
            shadow_target_symbol_geometry_order_equal=matches,
            shadow_launches=shadow_signature, target_launches=target_signature,
            binary_identity_proven=False))
        if not matches:
            row['isolation_reasons'].append('Native shadow and target symbols/geometry/order differ')
    row['status'] = 'complete' if all(item['status'] == 'complete'
        for records in row['native'].values() for item in records) else 'incomplete'
    row['cuda_allocated_bytes_at_case_end'] = torch.cuda.memory_allocated()
    aggregate(report)
    save(args.output, report)
    # Function scope releases every per-case input/cache/capture/closure. Prepared
    # models and their bounded module scratch stay alive across the four cases.


def child(args):
    global torch, reference, bench
    report = initial(args)
    started = time.monotonic()
    try:
        import torch
        import reference
        import compare_decode_public as bench
        bench.torch, bench.reference = torch, reference
        paths = dict(baseline=args.baseline, runner=__file__, helpers=bench.__file__, reference=reference.__file__,
            checks=Path(reference.__file__).with_name('checks.py'),
            nvfp4=Path(reference.__file__).with_name('nvfp4.py'))
        report['sources'] = {name: dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)
                             for name, path in paths.items()}
        save(args.output, report)
        if report['sources']['baseline']['sha256'] != BASE_SHA:
            raise RuntimeError('Baseline SHA mismatch; unchanged control is required')
        report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
            python=sys.version, cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'))
        modules = {role: bench.load(args.baseline, 'ceiling_' + role) for role in ('target', 'shadow')}
        weights = reference.generate_weights(seed=731283)
        states = {}
        report['prepare'] = {}
        for role in modules:
            own_weights = bench.clone_data(weights)
            before = time.monotonic()
            states[role] = modules[role].prepare(own_weights)
            torch.cuda.synchronize()
            report['prepare'][role] = dict(seconds=time.monotonic() - before,
                independent_cloned_model_weights=True,
                router_bias_unchanged=bytes_equal(own_weights['router_bias'], weights['router_bias']))
            del own_weights
            save(args.output, report)
        flush = torch.ones(FLUSH_BYTES // 4, dtype=torch.float32, device='cuda')
        report['flush'] = dict(bytes=FLUSH_BYTES, operation='read-only torch.sum; scalar sink retained by graph')
        for name, q_len in (('D1', 1), ('V1', 8)):
            for layer in WEIGHT:
                run_case(args, report, modules, states, weights, flush, name, q_len, layer)
                gc.collect()
                torch.cuda.empty_cache()
                report['cuda_allocated_after_case_cleanup'] = torch.cuda.memory_allocated()
                save(args.output, report)
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['status'] = 'complete' if all(row['status'] == 'complete' for row in report['results']) else 'incomplete'
    except BaseException as exc:
        report['status'] = 'failed'
        report['errors'].append(dict(error=repr(exc), traceback=traceback.format_exc()))
    finally:
        aggregate(report)
        report['elapsed_seconds'] = time.monotonic() - started
        save(args.output, report)
    return 0 if report['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='decode_cold_control.py')
    parser.add_argument('--output', required=True)
    parser.add_argument('--input-offset', type=int, default=0)
    parser.add_argument('--reps', type=int, default=12)
    parser.add_argument('--seconds', type=int, default=860, help='Hard child deadline; max 865, below 870s run cap')
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 865:
        parser.error('--seconds must be 1..865')
    if args.reps < 12 or args.reps % 4:
        parser.error('--reps must be a multiple of four and at least12')
    if args.child:
        return child(args)
    save(args.output, initial(args))
    prefix = Path(args.output).with_suffix('')
    logfile = str(prefix) + '-child.log'
    timed_out = False
    with open(logfile, 'w') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
            *sys.argv[1:], '--child'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    report = json.loads(Path(args.output).read_text())
    report.update(child_exit_code=code, parent_deadline_seconds=args.seconds)
    artifacts = [Path(logfile).name, *report.get('artifact_files', [])]
    # Recover a trace/raw artifact written immediately before a child deadline.
    artifacts.extend(path.name for path in prefix.parent.glob(prefix.name + '-*')
                     if path.is_file() and path.suffix in ('.json', '.pt', '.log'))
    report['artifact_files'] = list(dict.fromkeys(artifacts))
    if timed_out or code:
        report['status'] = 'timeout' if timed_out else 'failed'
        report['errors'].append(dict(error='Child deadline' if timed_out else 'Child exited unsuccessfully', code=code))
    aggregate(report)
    save(args.output, report)
    print(json.dumps(dict(status=report['status'], output=args.output, board=report['board_estimate'])), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
