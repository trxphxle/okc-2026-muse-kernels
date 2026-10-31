"""Collect portable actual-layer fixtures, then export18 matched expert tuples.

Run in the parent's bounded GPU sandbox (this script never allocates a sandbox):
  python round4_tuples.py --phase collect --baseline decode_round4_control.py
  python round4_tuples.py --phase profile --baseline decode_round4_control.py
Or use --phase all. Collection writes round4-fixtures.json and18 named PT files;
HOL can consume them before profiling. The cumulative active collection/profile
budget defaults to860s and cannot exceed860s. An incomplete run reports every
missing case and retains raw evidence; it never quietly narrows the fixture set.
Only duration and DRAM-read metrics are requested, application replay/cache none.
Multi-pass results are rejected rather than joining unmatched timing and bytes.
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

from round4_common import SOURCE_SHA, descriptors, sha, save
from round4_tuple_worker import frozen_manifest

METRICS = ('gpu__time_duration.sum', 'dram__bytes_read.sum')
PASS_METRICS = ('profiler__replayer_passes', 'profiler__replayer_passes_type_warmup')


def parse_csv(path):
    """Read raw-page wide CSV or details long CSV, retaining full launch keys."""
    records, order = {}, []
    header, units = None, None
    with path.open(newline='', errors='replace') as stream:
        for line_number, values in enumerate(csv.reader(stream), 1):
            columns = [v.strip().lstrip('\ufeff') for v in values]
            if 'ID' in columns and 'Kernel Name' in columns:
                header, units = columns, None
                continue
            if header is None:
                continue
            if len(values) != len(header):
                if values and values[0].strip().isdigit():
                    raise RuntimeError(f'NCU CSV malformed row{line_number}')
                continue
            row = dict(zip(header, values))
            identifier = row.get('ID', '').strip()
            if not identifier:
                units = row
                continue
            if not identifier.isdigit():
                continue
            identity = {key: row.get(key, '').strip() for key in
                ('ID', 'Process ID', 'Process Name', 'Host Name', 'Kernel Name',
                 'Context', 'Stream', 'Block Size', 'Grid Size')}
            key = tuple(identity[k] for k in ('Process ID', 'Context', 'Stream', 'ID'))
            if key not in records:
                records[key] = dict(launch_identity=identity, metrics={}, units={}, raw_values={},
                                    pass_metadata={}, csv_first_line=line_number)
                order.append(key)
            record = records[key]
            if identity != record['launch_identity']:
                raise RuntimeError('NCU metric rows disagree on launch identity')
            if 'Metric Name' in row:
                entries = [(row['Metric Name'], row.get('Metric Value', ''), row.get('Metric Unit', ''))]
            else:
                entries = [(name, row[name], (units or {}).get(name, ''))
                           for name in (*METRICS, *PASS_METRICS) if name in row]
            for name, raw, unit in entries:
                if name in PASS_METRICS:
                    value = int(raw.strip().replace(',', ''))
                    if name in record['pass_metadata']:
                        raise RuntimeError('Duplicate profiler pass metadata')
                    record['pass_metadata'][name] = value
                    continue
                if name not in METRICS:
                    continue
                if name in record['metrics']:
                    raise RuntimeError('Duplicate NCU metric for same launch: ' + name)
                try:
                    value = float(raw.strip().replace(',', '').replace('\u00a0', ''))
                except ValueError as exc:
                    raise RuntimeError('Missing/non-numeric NCU counter: ' + raw) from exc
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError('Invalid NCU counter: ' + raw)
                record['metrics'][name] = value
                record['units'][name] = unit.strip()
                record['raw_values'][name] = raw
    output = [records[key] for key in order]
    for record in output:
        if set(record['metrics']) != set(METRICS):
            raise RuntimeError('Duration/DRAM counters not on same launch')
        if record['pass_metadata'] != dict(zip(PASS_METRICS, (1, 0))):
            raise RuntimeError('Each launch must report exactly one collection pass and zero profiler warmup passes')
        if record['units'][METRICS[0]] not in ('nsecond', 'ns'):
            raise RuntimeError('Expected base nanoseconds, got ' + record['units'][METRICS[0]])
        if record['units'][METRICS[1]] not in ('byte', 'bytes', 'B'):
            raise RuntimeError('Expected base DRAM bytes, got ' + record['units'][METRICS[1]])
    return output


def worker_reports(prefix):
    prefix = Path(prefix)
    paths = sorted(prefix.parent.glob(prefix.name + '-pass-*.json'))
    return [(path, json.loads(path.read_text())) for path in paths]


def matched_tuples(manifest, report):
    workers = worker_reports(report['profile_worker_prefix'])
    report['profile_worker_reports'] = [str(path) for path, _ in workers]
    report['application_invocation_count'] = len(workers)
    # A repeated application would pair counters gathered on different launches;
    # preserve the evidence, but do not label those results matched same-pass.
    if len(workers) != 1:
        raise RuntimeError(f'Expected exactly one application metric pass, got {len(workers)} workers')
    path, worker = workers[0]
    if worker['status'] != 'complete' or worker.get('missing_case_ids'):
        raise RuntimeError('Profile worker incomplete: ' + str(path))
    if worker['source_sha256'] != SOURCE_SHA or worker['manifest_sha256'] != sha(Path(report['manifest'])):
        raise RuntimeError('Profile source/manifest drift')
    if worker['source_files'] != manifest['source_files']:
        raise RuntimeError('Collector/profile source dependencies differ')
    expected = [fixture['case_id'] for fixture in manifest['fixtures']]
    if [case['case_id'] for case in worker['cases']] != expected:
        raise RuntimeError('Profile case ordering/completeness mismatch')
    raw_text = '\n'.join(Path(report[key]).read_text(errors='replace') for key in ('csv', 'profile_log'))
    pass_counts = [int(x) for x in re.findall(r'\b(\d+)\s+pass(?:es)?\b', raw_text)]
    report['metric_pass_counts_from_raw_log'] = pass_counts
    report['profiler_diagnostics'] = [line for line in raw_text.splitlines()
        if line.startswith('==') and re.search(r'warn|error|pass|replay|serial', line, re.I)]
    if pass_counts and any(count != 1 for count in pass_counts):
        raise RuntimeError('Textual profiler log reports a multipass target')
    records = parse_csv(Path(report['csv']))
    report['raw_launch_count'] = len(records)
    if len(records) != len(expected):
        raise RuntimeError(f'Expected18 isolated expert launches, got {len(records)}')
    report['metric_pass_counts_from_csv'] = [r['pass_metadata'][PASS_METRICS[0]] for r in records]
    report['profiler_warmup_counts_from_csv'] = [r['pass_metadata'][PASS_METRICS[1]] for r in records]
    report['single_pass_evidence'] = 'One application invocation; each raw CSV launch has profiler__replayer_passes=1 and profiler__replayer_passes_type_warmup=0'
    if not Path(report['ncu_report']).is_file():
        raise RuntimeError('Raw NCU report absent')
    tuples = []
    for sequence, (fixture, case, launch) in enumerate(zip(manifest['fixtures'], worker['cases'], records)):
        if case['sequence'] != sequence or case['fixture_sha256'] != fixture['fixture_sha256']:
            raise RuntimeError('Fixture/launch sequence mismatch')
        if any(case[k] != 1 for k in
               ('profiler_regions_started', 'profiler_regions_completed', 'target_graph_replays')):
            raise RuntimeError('Not exactly one target replay per profiler region')
        if not all(check['passed'] for check in case['checks'].values()) or case['pdl'] != 0:
            raise RuntimeError('Profile correctness/PDL control failed')
        identity = launch['launch_identity']
        expected_big = '1' if fixture['m'] >= 64 else '0'
        if not re.search(r'moe_fused<(?:' + expected_big + ('|true' if expected_big == '1' else '|false') + r')>', identity['Kernel Name']):
            raise RuntimeError('Unexpected native expert variant in matched launch')
        if [int(x) for x in re.findall(r'\d+', identity['Grid Size'])] != [132, 1, 1]:
            raise RuntimeError('Unexpected native expert grid')
        if [int(x) for x in re.findall(r'\d+', identity['Block Size'])] != [576, 1, 1]:
            raise RuntimeError('Unexpected native expert block geometry')
        duration = launch['metrics'][METRICS[0]]
        dram = launch['metrics'][METRICS[1]]
        tuples.append(dict(**fixture, measured_dram_read_bytes=dram,
            measured_duration_ns=duration, matched_dram_read_GB_per_s=dram/duration,
            bandwidth_units='decimal GB/s = bytes/ns; same launch and same single metric pass only',
            timing_interval=case['profile_interval'], launch_identity=launch['launch_identity'],
            metric_pass_count=1, profiler_warmup_pass_count=0,
            pass_metadata=launch['pass_metadata'], application_pass=1, sequence=sequence,
            matching_method='All unfiltered NCU launches in execution order; worker asserts one native launch region per fixture; exact18 launch/18 region cardinality; one application invocation and one metric pass per launch',
            metric_units=launch['units'], raw_metric_values=launch['raw_values'],
            csv_first_line=launch['csv_first_line'],
            profile_controls=case, raw_csv=report['csv'], raw_log=report['profile_log'],
            raw_report=report['ncu_report'], worker_report=str(path)))
    return tuples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('collect', 'profile', 'all'), default='all')
    parser.add_argument('--baseline', type=Path, default=Path('decode_round4_control.py'))
    parser.add_argument('--bench', type=Path, default=Path('compare_decode_public.py'))
    parser.add_argument('--reference', type=Path, default=Path('reference.py'))
    parser.add_argument('--output', type=Path, default=Path('round4-tuples.json'))
    parser.add_argument('--seconds', type=float, default=860)
    parser.add_argument('--ncu', default='/usr/local/cuda/bin/ncu')
    parser.add_argument('--resume-report', type=Path)
    parser.add_argument('--parse-report', type=Path, help='Offline export from saved NCU artifacts; no GPU execution')
    args = parser.parse_args()
    if args.parse_report is not None:
        evidence_root = args.parse_report.resolve().parent
        report = json.loads(args.parse_report.read_text())
        report['parser_recovery'] = dict(original_report=str(args.parse_report),
            original_error=report.pop('error', None), parser_sha256=sha(Path(__file__)),
            reason='Use structured per-launch pass metadata, not optional progress text')
        for key in ('manifest', 'csv', 'profile_log', 'ncu_report', 'profile_worker_prefix'):
            report[key] = str(evidence_root / report[key])
        manifest = frozen_manifest(Path(report['manifest']))
        report['tuples'] = matched_tuples(manifest, report)
        report['campaign_complete'] = len(report['tuples']) == 18
        report['missing_tuple_case_ids'] = []
        report['all_AA_within_one_percent'] = all(row['unprofiled_paired_AA']['within_one_percent']
                                               for row in report['tuples'])
        report['status'] = 'complete'
        save(args.output, report)
        print(json.dumps(dict(status='complete', tuples=len(report['tuples']),
                              output=str(args.output), gpu_execution=False)))
        return 0
    if not 0 < args.seconds <= 860:
        parser.error('--seconds must be positive and at most860 (cleanup reserve below870)')
    if args.output.name != str(args.output) or args.output.suffix != '.json':
        parser.error('--output must be a JSON basename')
    started = time.monotonic()
    prefix = args.output.stem
    manifest_path = Path('round4-fixtures.json')
    report = dict(status='starting', phase=args.phase, source_sha256=SOURCE_SHA,
        submission_allowed=False, candidate_built=False, tuples=[], commands=[], artifact_files=[],
        manifest=str(manifest_path), requested_case_ids=[d['case_id'] for d in descriptors()],
        collection_worker_prefix=prefix+'-collect', profile_worker_prefix=prefix+'-profile',
        collection_log=prefix+'-collect.log', profile_log=prefix+'-ncu.log',
        csv=prefix+'-ncu.csv', ncu_report=prefix+'-ncu.ncu-rep',
        ncu_version_file=prefix+'-ncu-version.txt', active_budget_seconds=args.seconds,
        budget_note='Collection plus profiling active elapsed; intervening parent HOL time excluded. Hard process-group deadline, no automatic retry.',
        metric_names=list(METRICS), cache_control='none', replay_mode='application',
        unprofiled_events_policy='Separate paired A/A only; never a DRAM bandwidth denominator')
    prior_seconds = 0.0
    if args.resume_report is not None:
        previous = json.loads(args.resume_report.read_text())
        prior_seconds = previous.get('prior_collection_seconds', 0.0) + previous['elapsed_seconds']

    def checkpoint():
        names = {args.output.name}
        names.update(str(p) for p in Path('.').glob(prefix+'-*') if p.is_file())
        if manifest_path.exists():
            names.add(manifest_path.name)
        names.update(d['fixture_file'] for d in descriptors() if Path(d['fixture_file']).is_file())
        report['artifact_files'] = sorted(names)
        report['elapsed_seconds'] = time.monotonic() - started
        report['cumulative_active_seconds'] = prior_seconds + report['elapsed_seconds']
        complete = {row['case_id'] for row in report['tuples']}
        report['missing_tuple_case_ids'] = [case for case in report['requested_case_ids'] if case not in complete]
        save(args.output, report)

    def run(command, log):
        remaining = args.seconds - prior_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Round4 active collection/profile deadline exhausted')
        entry = dict(command=command, log=log, remaining_seconds=remaining)
        report['commands'].append(entry)
        checkpoint()
        with Path(log).open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                entry['returncode'] = process.wait(timeout=remaining)
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
            raise RuntimeError(f'{log}: exit{entry["returncode"]}; evidence preserved')

    worker_base = [sys.executable, str(Path(__file__).with_name('round4_tuple_worker.py')),
        '--baseline', str(args.baseline), '--bench', str(args.bench), '--reference', str(args.reference),
        '--manifest', str(manifest_path)]
    if args.resume_report is not None:
        if args.phase != 'collect':
            parser.error('--resume-report applies only to explicit collection recovery')
        worker_base += ['--resume-report', str(args.resume_report)]
    try:
        if sha(args.baseline) != SOURCE_SHA:
            raise RuntimeError('Qualified decoder SHA mismatch')
        if args.phase in ('collect', 'all'):
            if worker_reports(report['collection_worker_prefix']):
                raise RuntimeError('Collection evidence already exists; use --phase profile or a fresh directory')
            report['status'] = 'collecting'
            run(worker_base + ['--phase', 'collect', '--output-prefix', report['collection_worker_prefix']],
                report['collection_log'])
            collection = worker_reports(report['collection_worker_prefix'])
            if len(collection) != 1 or collection[0][1]['status'] != 'complete':
                raise RuntimeError('Collector did not complete exactly once')
        manifest = frozen_manifest(manifest_path)
        report['manifest_sha256'] = sha(manifest_path)
        report['collection_seconds'] = manifest['collection_seconds']
        report['fixture_count'] = len(manifest['fixtures'])
        report['missing_fixture_case_ids'] = []
        if args.phase == 'profile':
            prior_seconds = manifest['collection_seconds']
        if args.phase == 'collect':
            report['status'] = 'collected'
            report['campaign_complete'] = False
        else:
            if worker_reports(report['profile_worker_prefix']) or Path(report['ncu_report']).exists():
                raise RuntimeError('Profile evidence already exists; use a distinct --output prefix')
            report['status'] = 'profiling'
            run([args.ncu, '--version'], report['ncu_version_file'])
            command = [args.ncu, '--profile-from-start', 'off', '--clock-control', 'none',
                '--cache-control', 'none', '--replay-mode', 'application', '--app-replay-mode', 'strict',
                '--app-replay-match', 'grid', '--disable-extra-suffixes', '--pipeline-boost-state', 'dynamic',
                '--graph-profiling', 'node', '--csv', '--page', 'raw', '--print-units', 'base',
                '--kernel-name-base', 'mangled', '--metrics', ','.join(METRICS),
                '--log-file', report['csv'], '--export', report['ncu_report']]
            run(command + worker_base + ['--phase', 'profile', '--output-prefix', report['profile_worker_prefix']],
                report['profile_log'])
            report['tuples'] = matched_tuples(manifest, report)
            report['campaign_complete'] = len(report['tuples']) == 18
            report['all_AA_within_one_percent'] = all(
                row['profile_controls']['unprofiled_paired_AA']['within_one_percent'] for row in report['tuples'])
            report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', campaign_complete=False,
            error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
    finally:
        if manifest_path.exists():
            try:
                partial = json.loads(manifest_path.read_text())
                collected = {d['case_id'] for d in partial.get('fixtures', [])}
                report['missing_fixture_case_ids'] = [d['case_id'] for d in descriptors() if d['case_id'] not in collected]
            except (ValueError, KeyError):
                report['missing_fixture_case_ids'] = report['requested_case_ids']
        else:
            report['missing_fixture_case_ids'] = report['requested_case_ids']
        checkpoint()
        print(json.dumps(dict(result=args.output.name, status=report['status'],
                              tuples=len(report['tuples']))), flush=True)
    return 0 if report['status'] in ('complete', 'collected') else 1


if __name__ == '__main__':
    sys.exit(main())
