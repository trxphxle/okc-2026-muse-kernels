"""Bounded D1/V1 cold-graph instruction-fetch diagnostic, never a submission.

Upload cold_ncu.py, cold_ncu_worker.py, decode_cold_control.py,
compare_decode_public.py, reference.py, checks.py and nvfp4.py to /workspace.
Run (default requests ALL FOUR cold cases, with a shared 860-second deadline):
  python cold_ncu.py --baseline decode_cold_control.py --output cold-ncu.json
Recommended bounded split, repeating explicitly for D1-full/V1-sliding/V1-full:
  python cold_ncu.py --baseline decode_cold_control.py --case D1-sliding \
    --output cold-ncu-D1-sliding.json
Use --metric-set instruction for an initial duration/cycles/no_instruction +
available instruction-cache subset. It is explicitly INCOMPLETE for the full
DRAM/long-scoreboard/register deliverable. Full is the default. --parse-only
reparses a saved output manifest without CUDA/NCU. A unique output basename is
required for each invocation; partial artifacts are retained, never overwritten.

Application replay recreates prepare, capture, warmups and one natural whole-
layer flush on every pass. --cache-control none is mandatory: there is NO flush
before individual child kernels. Per-kernel NCU may nevertheless serialize PDL,
change scheduling, perturb instruction caches and impose replay overhead. These
counters are profiler evidence, not native first-touch latency or additive
latency savings. Pair with separate native cold/warm timing. --cache-mode warm
is optional combined-code/data-cache evidence, not a legal optimization.

The four cases are D1/V1 x sliding/full; no timeout silently reduces that scope.
Split reports mark outstanding cases and never claim full campaign completion.
The raw decoder is hash-locked and precision-invalid for submission even if its
public numerical oracle passes. All source/notices remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

from cold_ncu_worker import CASES, SOURCE_SHA, save, sha

NCU = '/usr/local/cuda/bin/ncu'
GROUPS = {
    'duration': ('gpu__time_duration.sum',),
    'no_instruction': (
        'smsp__warp_issue_stalled_no_instruction_per_warp_active.pct',
        'smsp__warp_issue_stalled_no_instructions_per_warp_active.pct',
        'smsp__warps_issue_stalled_no_instruction.sum',
        'smsp__warps_issue_stalled_no_instructions.sum'),
    'sm_cycles': ('sm__cycles_elapsed.avg', 'sm__cycles_elapsed.sum'),
    'dram_read': ('dram__bytes_read.sum',),
    'dram_write': ('dram__bytes_write.sum',),
    'long_scoreboard': ('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
                        'smsp__warps_issue_stalled_long_scoreboard.sum'),
    'registers': ('launch__registers_per_thread',),
    'immediate_constant_stall': ('smsp__warp_issue_stalled_imc_miss_per_warp_active.pct',),
}
ICACHE_NAMES = (
    'sm__icc_requests.sum', 'sm__icc_requests_lookup_hit.sum',
    'sm__icc_requests_lookup_miss.sum', 'sm__icc_requests_hit.sum',
    'sm__icc_requests_miss.sum', 'sm__icc_request_hit_rate.pct',
    'smsp__icc_requests.sum', 'smsp__icc_requests_lookup_miss.sum',
    'sm__icache_requests.sum', 'sm__icache_requests_lookup_miss.sum',
    'smsp__icache_requests.sum', 'smsp__icache_requests_lookup_miss.sum',
)
INSTRUCTION_GROUPS = ('duration', 'no_instruction', 'sm_cycles')
CAVEATS = [
    'NCU application replay + cache-control none; never kernel-replay cache-control all.',
    'The 256MiB read flush is outside the sole CUDA-profiler range and precedes the whole layer, not each child.',
    'A data/L2 flush is not proof that every instruction-cache level starts cold.',
    'Application replay repeats baseline prepare/tuning; strict matching and fixture/mode manifests detect drift.',
    'NCU node profiling may serialize PDL and alter cache/SM residency; reported order is launch order, not a native critical path.',
    'no_instruction can include instruction-buffer/fetch unavailability; it is not exclusively an instruction-cache miss measurement.',
    'imc_miss is immediate-constant-cache waiting, NOT an instruction-cache miss counter.',
    'Stall percentages, SM-cycle counts, cache misses and kernel durations are not additive removable latency.',
    'Warm mode warms both code and data; no code-only causal attribution or optimization claim.',
    'Raw baseline lacks mandatory BF16 boundaries; numerical tolerance passing is not precision legality.',
    'No board speedup is computed: all six loads are equal; small-T hypotheses price D16/D32/V16/V32 at ratio1 and layers at46:16.',
]


def inventory(path):
    """Stream the potentially very large NCU --query-metrics-mode all table."""
    names, relevant = set(), []
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            fields = line.strip().split(None, 1)
            if not fields:
                continue
            token = fields[0].strip('"')
            if re.fullmatch(r'[a-z][a-z0-9_]*__[A-Za-z0-9_.]+', token):
                names.add(token)
                if ('no_instruction' in token or 'stalled_imc_miss' in token
                        or re.search(r'__(?:icc|icache|instruction_cache)_', token)):
                    relevant.append(line.rstrip())
    return names, relevant


def choose_metrics(available, metric_set):
    group_selection = {name: next((m for m in alternatives if m in available), None)
                       for name, alternatives in GROUPS.items()}
    # --query-metrics enumerates hardware metrics, not launch metadata on NCU
    # 2025.3. Request registers explicitly and require a numeric result in CSV.
    group_selection['registers'] = 'launch__registers_per_thread'
    # Only named instruction-cache request/lookup/miss/hit counters, not generic
    # L1/L2 hit rates, count as instruction-cache evidence. Do not substitute PCSAMP
    # samples for the mandatory scheduler no_instruction metric silently.
    cache_candidates = sorted(m for m in available
        if re.match(r'^(?:sm|smsp)__', m)
        and re.search(r'__(?:icc|icache|instruction_cache)_', m)
        and re.search(r'request|lookup|miss|hit', m)
        and (m.endswith('.sum') or m.endswith('_rate.pct')))
    selected_groups = INSTRUCTION_GROUPS if metric_set == 'instruction' else tuple(GROUPS)
    selected = [group_selection[g] for g in selected_groups if group_selection[g]]
    selected.extend(m for m in cache_candidates if m not in selected)
    requested = [m for alternatives in GROUPS.values() for m in alternatives] + list(ICACHE_NAMES)
    return dict(selected_metrics=selected, metric_groups=group_selection,
        required_groups=list(selected_groups),
        missing_required_groups=[g for g in selected_groups if not group_selection[g]],
        unavailable_requested_names=[m for m in requested if m not in available and not m.startswith('launch__')],
        nonenumerated_launch_metrics=[m for m in GROUPS['registers'] if m not in available],
        launch_metadata_policy='Explicitly requested; availability is established only by the actual numeric CSV result',
        available_instruction_cache_metrics=cache_candidates,
        instruction_cache_counter_status='available' if cache_candidates else 'unavailable_in_installed_query',
        deferred_full_groups=[g for g in GROUPS if g not in selected_groups],
        full_group_request_resolved=all(group_selection.values()))


def numeric(raw):
    text = raw.strip().replace(',', '').replace('\u00a0', '')
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def parse_csv(path, selected):
    """Accept both NCU details (long) and raw-page (wide + units row) CSV.

    Preserve raw symbols (mangled, demangled, and qualified spellings alike),
    units and invalid values. A missing/N/A required value is an error, not zero.
    Parsing problems return alongside partial records so evidence is still saved.
    """
    records, order, errors = {}, [], []
    header, units, wide = None, None, False
    selected = set(selected)
    with Path(path).open(newline='', errors='replace') as stream:
        for line_number, values in enumerate(csv.reader(stream), 1):
            if not values:
                continue
            columns = [v.strip().lstrip('\ufeff') for v in values]
            if 'ID' in columns and any(k in columns for k in ('Kernel Name', 'Kernel', 'Name')):
                header = columns
                wide = 'Metric Name' not in header
                units = None
                continue
            if header is None:
                continue
            if len(values) != len(header):
                if values[0].strip().isdigit():
                    errors.append(f'CSV row {line_number}: wrong column count')
                continue
            row = dict(zip(header, values))
            if wide and not row.get('ID', '').strip():
                units = row
                continue
            identifier = row.get('ID', '').strip()
            if not identifier.isdigit():
                continue
            symbol = next((row[k].strip() for k in ('Kernel Name', 'Kernel', 'Name') if row.get(k)), '')
            if not symbol:
                errors.append(f'CSV row {line_number}: empty kernel symbol')
                continue
            if identifier not in records:
                records[identifier] = dict(id=identifier, kernel=symbol, metrics={}, units={}, raw_values={},
                    process_id=row.get('Process ID'), context=row.get('Context'), stream=row.get('Stream'),
                    block_size=row.get('Block Size'), grid_size=row.get('Grid Size'),
                    csv_first_line=line_number)
                order.append(identifier)
            record = records[identifier]
            if record['kernel'] != symbol or record['process_id'] != row.get('Process ID'):
                errors.append(f'CSV ID {identifier}: inconsistent kernel/process identity')
            if wide:
                metrics = [(m, row[m], (units or {}).get(m, '')) for m in selected if m in row]
            else:
                metric = row.get('Metric Name', '').strip()
                metrics = [(metric, row.get('Metric Value', ''), row.get('Metric Unit', ''))] if metric in selected else []
            for metric, raw, unit in metrics:
                if metric in record['raw_values']:
                    errors.append(f'CSV ID {identifier}: duplicate metric {metric}')
                    continue
                record['raw_values'][metric] = raw
                record['units'][metric] = unit.strip()
                value = numeric(raw)
                if value is None:
                    errors.append(f'CSV ID {identifier}: invalid/unavailable {metric}={raw!r}')
                else:
                    record['metrics'][metric] = value
    if header is None:
        errors.append('NCU CSV metric header missing')
    if [int(k) for k in order] != sorted(set(int(k) for k in order)):
        errors.append('NCU launch IDs are not uniquely ordered')
    for index, record in enumerate(records.values()):
        record['launch_index'] = index
        missing = sorted(selected - record['metrics'].keys())
        record['missing_selected_metrics'] = missing
        if missing:
            errors.append(f'CSV ID {record["id"]}: missing selected counters {missing}')
        for metric, unit in record['units'].items():
            if metric == 'gpu__time_duration.sum':
                factors = {'ns': .001, 'nsecond': .001, 'nseconds': .001,
                           'us': 1., 'usecond': 1., 'useconds': 1., 'µs': 1.,
                           'ms': 1000., 'msecond': 1000., 's': 1e6, 'second': 1e6}
                if unit not in factors:
                    errors.append(f'CSV ID {record["id"]}: unsupported duration unit {unit!r}')
                elif metric in record['metrics']:
                    record['duration_us'] = record['metrics'][metric] * factors[unit]
                    if record['duration_us'] <= 0:
                        errors.append(f'CSV ID {record["id"]}: zero duration')
            elif metric.startswith('dram__bytes_') and unit not in ('byte', 'bytes', 'B'):
                errors.append(f'CSV ID {record["id"]}: non-base byte unit {unit!r}')
            elif metric.endswith('.pct') and unit not in ('%', 'pct', 'percent'):
                errors.append(f'CSV ID {record["id"]}: unexpected percentage unit {unit!r}')
            elif '__cycles_' in metric and unit not in ('cycle', 'cycles'):
                errors.append(f'CSV ID {record["id"]}: unexpected cycle unit {unit!r}')
    if not records:
        errors.append('No per-kernel native records; graph aggregate is not sufficient')
    return list(records.values()), errors


def worker_reports(case):
    paths = sorted(Path('.').glob(case['worker_prefix'] + '-pass-*.json'))
    reports, errors = [], []
    for path in paths:
        try:
            report = json.loads(path.read_text())
            reports.append(dict(path=path.name, report=report))
        except Exception as exc:
            errors.append(f'{path.name}: {exc}')
    return reports, errors


def merge_case(case, report):
    workers, errors = worker_reports(case)
    case['worker_passes'] = workers
    if not workers:
        errors.append('No worker invocation metadata found')
    stable = None
    for item in workers:
        child = item['report']
        if child.get('status') != 'complete':
            errors.append(item['path'] + ': incomplete worker')
        if (child.get('case_id') != case['case_id'] or child.get('source_sha256') != SOURCE_SHA
                or child.get('cache_mode') != report['cache_mode']):
            errors.append(item['path'] + ': case/source/cache mismatch')
        if any(child.get(k) != 1 for k in ('profiler_regions_started', 'profiler_regions_completed', 'target_graph_replays')):
            errors.append(item['path'] + ': expected exactly one region/target graph')
        if child.get('capture', {}).get('includes_flush') is not False:
            errors.append(item['path'] + ': capture flush contract missing/incorrect')
        # Strict NCU name/grid/block/order matching checks the selected graph.
        # Retain complete prepare modes, but unrelated T16/T32/T128/T256 tuner
        # choices must not reject a deterministic T1/T8 target.
        signature = {k: child.get(k) for k in ('sources', 'shape', 'input_seed', 'input_sha256',
            'weight_seed', 'input_offset', 'local_start', 'baseline_probe_shard_counts', 'routes')}
        if stable is None:
            stable = signature
        elif stable != signature:
            errors.append(item['path'] + ': application replay fixture/source/routes drift')
    if Path(case['csv']).exists():
        records, parse_errors = parse_csv(case['csv'], report['selected_metrics'])
        case['kernels'] = records
        errors.extend(parse_errors)
        groups = report['metric_groups']
        for record in records:
            record['case_id'] = case['case_id']
            record['evidence'] = {group: dict(metric=metric,
                value=record['metrics'].get(metric), unit=record['units'].get(metric))
                for group, metric in groups.items() if metric in report['selected_metrics']}
            record['instruction_cache_evidence'] = {metric: dict(value=record['metrics'].get(metric),
                unit=record['units'].get(metric)) for metric in report['available_instruction_cache_metrics']}
    else:
        errors.append('NCU CSV missing')
    modes = [item['report'].get('prepare', {}).get('mode') for item in workers]
    case['all_prepare_modes_equal'] = bool(modes) and all(mode == modes[0] for mode in modes[1:])
    case['ncu_diagnostic_lines'] = []
    for filename in (case['stdout'], case['csv']):
        if Path(filename).exists():
            with Path(filename).open(errors='replace') as stream:
                for line in stream:
                    if line.startswith('==') and re.search(r'warn|error|serial|depend|replay|pass', line, re.I):
                        case['ncu_diagnostic_lines'].append(dict(file=filename, text=line.rstrip()))
    case['numerically_valid'] = bool(workers) and all(x['report'].get('oracle', {}).get('passed') is True for x in workers)
    case['errors'] = errors
    case['status'] = 'failed' if errors else ('complete' if case['numerically_valid'] else 'complete_with_oracle_failure')
    case['attribution'] = ('Separate NCU application for this case, one CUDA-profiler region containing exactly one unchanged graph per pass; '
        'all unfiltered node records ordered by native launch ID. Raw symbols retained, no substring stage guess or timestamp containment.')
    return not errors


def artifacts(report):
    names = set(report.get('inventory_files', {}).values())
    for case in report['cases']:
        names.update(case[k] for k in ('csv', 'stdout', 'ncu_report'))
        for path in Path('.').glob(case['worker_prefix'] + '-pass-*'):
            if path.suffix in ('.json', '.pt'):
                names.add(path.name)
    return sorted(name for name in names if Path(name).is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--bench', type=Path, default=Path('compare_decode_public.py'))
    parser.add_argument('--reference', type=Path, default=Path('reference.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--cache-mode', choices=('cold', 'warm'), default='cold')
    parser.add_argument('--metric-set', choices=('instruction', 'full'), default='full')
    parser.add_argument('--input-offset', type=int, default=1)
    parser.add_argument('--seconds', type=int, default=860)
    parser.add_argument('--ncu', default=NCU)
    parser.add_argument('--parse-only', action='store_true')
    args = parser.parse_args()
    if args.output.name != str(args.output) or args.output.suffix != '.json':
        parser.error('--output must be a JSON basename for run_layer retrieval')
    if not 1 <= args.seconds <= 860:
        parser.error('--seconds must be 1..860 (ten seconds reserved below the870s cap)')
    if args.output.exists() and not args.parse_only:
        parser.error('Output already exists; choose a unique basename to retain prior evidence')
    started = time.monotonic()
    prefix = args.output.stem
    selected_cases = [args.case] if args.case else list(CASES)
    report = dict(status='starting', kind='unchanged_decode_instruction_fetch_diagnostic',
        official_score=False, submission_allowed=False, source_sha256=SOURCE_SHA,
        requested_cases=selected_cases, campaign_cases=list(CASES), cache_mode=args.cache_mode,
        metric_set=args.metric_set, deadline_seconds=args.seconds,
        scope_complete=False, campaign_complete=False, artifact_files=[], caveats=CAVEATS,
        replay_policy=dict(mode='application', matching='strict name/grid/block/order',
            graph_profiling='node', cache_control='none', clock_control='none',
            pipeline_boost_state='dynamic', kernel_filter=None, launch_count=None),
        inventory_files={k: prefix + suffix for k, suffix in {
            'version': '-ncu-version.txt', 'help': '-ncu-help.txt', 'metrics': '-ncu-metrics.txt'}.items()},
        commands=[], cases=[dict(case_id=case_id, status='not_run',
            csv=f'{prefix}-{case_id}.csv', stdout=f'{prefix}-{case_id}.log',
            ncu_report=f'{prefix}-{case_id}.ncu-rep', worker_prefix=f'{prefix}-{case_id}-worker')
            for case_id in selected_cases])

    def checkpoint():
        report['artifact_files'] = artifacts(report)
        report['elapsed_seconds'] = time.monotonic() - started
        save(args.output, report)

    def run(command, logfile, cap=None):
        remaining = args.seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Total profile deadline exhausted; outstanding cases are not complete')
        entry = dict(command=command, stdout=logfile, budget_seconds=min(remaining, cap) if cap else remaining)
        report['commands'].append(entry)
        checkpoint()
        print(json.dumps(entry), flush=True)
        with Path(logfile).open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                entry['returncode'] = process.wait(timeout=entry['budget_seconds'])
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                entry.update(returncode=process.returncode, interrupted=True)
                raise
        checkpoint()
        if entry['returncode']:
            raise RuntimeError(f'{logfile}: exit{entry["returncode"]}; raw evidence preserved')

    try:
        if args.parse_only:
            report = json.loads(args.output.read_text())
            report['status'] = 'reparsing'
            report['parse_only'] = True
        else:
            checkpoint()
            if sha(args.baseline) != SOURCE_SHA:
                raise RuntimeError('Baseline SHA mismatch; refusing changed decoder')
            report['sources'] = {role: dict(path=str(path), sha256=sha(path)) for role, path in {
                'baseline': args.baseline, 'runner': Path(__file__),
                'worker': Path(__file__).with_name('cold_ncu_worker.py'),
                'bench': args.bench, 'reference': args.reference,
                'checks': args.reference.with_name('checks.py'),
                'nvfp4': args.reference.with_name('nvfp4.py')}.items()}
            report['status'] = 'querying_installed_ncu'
            run([args.ncu, '--version'], report['inventory_files']['version'], 15)
            run([args.ncu, '--help'], report['inventory_files']['help'], 15)
            help_text = Path(report['inventory_files']['help']).read_text(errors='replace')
            needed_flags = ('--profile-from-start', '--clock-control', '--cache-control', '--replay-mode',
                '--graph-profiling', '--csv', '--page', '--print-units', '--log-file', '--metrics',
                '--kernel-name-base', '--export', '--app-replay-mode', '--app-replay-match',
                '--disable-extra-suffixes', '--pipeline-boost-state')
            report['missing_required_flags'] = [flag for flag in needed_flags if flag not in help_text]
            if report['missing_required_flags']:
                raise RuntimeError('Installed NCU lacks required flags: ' + repr(report['missing_required_flags']))
            run([args.ncu, '--query-metrics', '--query-metrics-mode', 'all'], report['inventory_files']['metrics'], 45)
            available, relevant = inventory(report['inventory_files']['metrics'])
            report.update(choose_metrics(available, args.metric_set))
            report['installed_instruction_metric_descriptions'] = relevant
            checkpoint()
            if report['missing_required_groups']:
                raise RuntimeError('Missing required installed metric groups: ' + repr(report['missing_required_groups']))
        for case in report['cases']:
            if not args.parse_only:
                report['status'] = 'profiling'
                case['status'] = 'running'
                checkpoint()
                command = [args.ncu, '--profile-from-start', 'off', '--clock-control', 'none',
                    '--cache-control', 'none', '--replay-mode', 'application', '--app-replay-mode', 'strict',
                    '--app-replay-match', 'grid', '--disable-extra-suffixes',
                    '--pipeline-boost-state', 'dynamic',
                    '--graph-profiling', 'node', '--csv', '--page', 'raw', '--print-units', 'base',
                    '--kernel-name-base', 'mangled', '--metrics', ','.join(report['selected_metrics']),
                    '--log-file', case['csv'], '--export', case['ncu_report'],
                    sys.executable, str(Path(__file__).with_name('cold_ncu_worker.py')),
                    '--baseline', str(args.baseline), '--bench', str(args.bench), '--reference', str(args.reference),
                    '--case', case['case_id'], '--cache-mode', args.cache_mode,
                    '--input-offset', str(args.input_offset), '--output', case['worker_prefix'] + '.json']
                # No kernel-name filter / launch-count: retain all graph children
                # and any unexpected launches instead of hiding attribution errors.
                run(command, case['stdout'])
            ok = merge_case(case, report)
            checkpoint()
            if not ok:
                raise RuntimeError(case['case_id'] + ': incomplete/mismatched profile evidence; see case errors')
        report['scope_complete'] = all(case['status'].startswith('complete') for case in report['cases'])
        report['all_oracles_passed'] = all(case['numerically_valid'] for case in report['cases'])
        report['outstanding_campaign_cases'] = [case for case in CASES if case not in report['requested_cases']]
        report['campaign_complete'] = bool(report['scope_complete'] and not report['outstanding_campaign_cases']
            and report['metric_set'] == 'full' and report['cache_mode'] == 'cold')
        report['status'] = 'complete' if report['all_oracles_passed'] else 'complete_with_oracle_failure'
    except BaseException as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        # Preserve partial parsed counters/metadata even when NCU exits or times out.
        for case in report['cases']:
            if case['status'] == 'running':
                try:
                    merge_case(case, report)
                    case['status'] = 'failed'
                except Exception as merge_error:
                    case['partial_parse_error'] = repr(merge_error)
    finally:
        checkpoint()
        print(json.dumps(dict(output=str(args.output), status=report['status'],
                              campaign_complete=report.get('campaign_complete', False))), flush=True)
    return 0 if report['status'].startswith('complete') else 1


if __name__ == '__main__':
    sys.exit(main())
