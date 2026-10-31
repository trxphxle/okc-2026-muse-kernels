"""One bounded Modal H200 lab.

The user removed all Modal spending caps. Billing and bounded lifetime estimates
remain visible before each allocation, but they are not authorization limits.
Historical reservations are bookkeeping, not charges.
No deployed functions, persistent volumes, secrets, or network ingress.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import math
import os
import subprocess
import time
import modal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/openkernel-challenge'
# Parallel lanes (COORDINATION.md): OKC_LANE=<name> gives a lane its own ledger,
# sandbox and output directories so lanes never reuse or stop each other's GPU.
LANE = os.environ.get('OKC_LANE', '').strip()
MAIN_LEDGER = OUT / 'modal-budget.json'
LEDGER = OUT / f'modal-budget-{LANE}.json' if LANE else MAIN_LEDGER
# Active Modal profile (= workspace). Sandboxes and images are only reused within it.
WORKSPACE = modal.config._profile
APP = 'okc-trxphxle-01a18750'
IMAGE = 'lmsysorg/sglang:dev-muse-spark-open'
IMAGE_BUILD_HEADROOM_USD = 5.0
TIMEOUT = 1800
CPU = 16.0  # Parallel native compilation for variant batches.
MEMORY = 32768
# Published GPU rate plus a conservative3x multiplier for CPU/memory.
UPPER_RATE = 0.001261 + 3 * (CPU * 0.0000131 + (MEMORY / 1024) * 0.00000222)
CANDIDATE = ROOT / 'archive' / OUT.relative_to(ROOT) / 'moe/submission_wgs_wide128_two_wg.py'
SOURCE_HASH = 'fc659d170494b3e88702776368122291be01d7ab46b9a55e764a41be51d1dd89'
PROBE = ROOT / 'work/modal/resource_probe.py'
COMPARE = ROOT / 'work/modal/synthetic_compare.py'
COMPARE_HASH = 'd0bd1b0a2177d46d4ffd8fcfb2a041f892bcda2c4c884ca17af7885ea163bce8'
NVFP4 = OUT / 'moe/nvfp4.py'
NVFP4_HASH = 'cfdf37c7bc14b10f7353aba7d4632f4767401b92b4b073941a8030d9a961aaaf'
COUNTER = ROOT / 'work/modal/counter_probe.py'
COUNTER_HASH = '7a14e64926bd8ec6afd0d84bffb7948756ae0ae25a19e080cf3da388d6d346af'
NCU_RUNNER = ROOT / 'work/modal/ncu_runner.py'
NCU_RUNNER_HASH = '15735d87576910ff7798484e47f698cbdaa5a63cbb20948a9090a0a8faa7f1d1'
NCU_FILTER = 'regex:_ZN3okc9moe_fusedILb[01]EEEvNS_6ParamsE'

def now():
    return datetime.now(timezone.utc).isoformat()

def read():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    d = dict(authorized_usd=None, authorized_purpose='Open Kernel Challenge GPU development',
             pricing_url='https://modal.com/pricing', pricing_checked_at=now(),
             conservative_rate_per_second=UPPER_RATE, image_builds=[], sessions=[])
    if LANE and MAIN_LEDGER.exists():
        # A new lane starts from the main ledger's current image instead of rebuilding.
        main = json.loads(MAIN_LEDGER.read_text())
        d['image_id'] = main.get('image_id')
        d['image_ids'] = main.get('image_ids', {})
    return d

def save(d):
    d['updated_at'] = now()
    # Retain historical bookkeeping without using it as the spending gate.
    d['reserved_usd'] = sum(x.get('reserved_usd', 0.0) for x in d['image_builds'] + d['sessions'])
    tmp = LEDGER.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=2) + '\n')
    tmp.replace(LEDGER)

def allocation_seconds():
    checkpoint = ROOT / 'work/active_checkpoint.json'
    if not checkpoint.exists():
        return TIMEOUT
    campaign = json.loads(checkpoint.read_text()).get('expert_campaign', {})
    if not campaign.get('approved') or not campaign.get('deadline_utc'):
        return TIMEOUT
    deadline = datetime.fromisoformat(campaign['deadline_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise RuntimeError('Expert campaign deadline must include a timezone')
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds()) - 60
    if remaining <= 0:
        raise RuntimeError('Expert campaign deadline reached or too close for another bounded allocation')
    return min(TIMEOUT, remaining)


def billing_snapshot(d, proposed_headroom):
    checked = datetime.now(timezone.utc)
    # Hourly reports exclude the open hour. Recent finished sessions use an
    # observed-lifetime upper bound; a live/unknown session uses its full cap.
    covered_before = checked.replace(minute=0, second=0, microsecond=0)
    starts = [x['requested_at'] for x in d['image_builds'] + d['sessions'] if x.get('requested_at')]
    first_month = min(starts or [checked.isoformat()])[:7]
    year, month = map(int, first_month.split('-'))
    cycles = {}
    while (year, month) <= (checked.year, checked.month):
        cycle = f'{year:04d}-{month:02d}'
        raw = subprocess.check_output(
            ['modal', 'billing', 'summary', '--for', cycle, '--json'], text=True, timeout=60)
        summary = json.loads(raw)
        metered, billed = float(summary['metered_cost']), float(summary['billed_cost'])
        if not all(math.isfinite(v) and v >= 0 for v in (metered, billed)):
            raise RuntimeError('Modal billing returned invalid costs')
        cycles[cycle] = dict(metered_usd=metered, billed_usd=billed, summary=summary)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    resources = json.loads(subprocess.check_output(
        ['modal', 'billing', 'report', '--for', 'today', '-r', 'h', '--show-resources', '--json'],
        text=True, timeout=60))
    unreported = 0.0
    for entry in d['sessions']:
        if not entry.get('requested_at'):
            continue
        began = datetime.fromisoformat(entry['requested_at'].replace('Z', '+00:00'))
        ended_text = entry.get('observed_finished_at')
        ended = datetime.fromisoformat(ended_text.replace('Z', '+00:00')) if ended_text else None
        if entry.get('state') == 'finished' and ended is not None and ended < covered_before:
            continue
        seconds = float(entry.get('timeout_seconds', TIMEOUT))
        if entry.get('state') == 'finished' and ended is not None:
            seconds = min(seconds, max(0.0, (ended - began).total_seconds()))
        unreported += seconds * UPPER_RATE
    for entry in d['image_builds']:
        ended = entry.get('completed_at')
        if entry.get('state') != 'built' or not ended or datetime.fromisoformat(ended) >= covered_before:
            unreported += IMAGE_BUILD_HEADROOM_USD
    metered = sum(v['metered_usd'] for v in cycles.values())
    billed = sum(v['billed_usd'] for v in cycles.values())
    snapshot = dict(checked_at=checked.isoformat(), cycles=cycles,
                    hourly_resources=resources, full_hours_before=covered_before.isoformat(),
                    metered_usd=metered, billed_usd=billed,
                    unreported_upper_usd=unreported, proposed_headroom_usd=proposed_headroom,
                    projected_upper_usd=max(metered, billed) + unreported + proposed_headroom,
                    cap_usd=None, spend_policy=None,
                    basis='Reported actual costs plus current-hour lifetime estimates; not a spending gate')
    d['billing'] = snapshot
    save(d)
    return snapshot


def budget():
    d = read()
    snapshot = billing_snapshot(d, allocation_seconds() * UPPER_RATE)
    print(json.dumps({k: snapshot[k] for k in
        ('checked_at', 'metered_usd', 'billed_usd', 'unreported_upper_usd',
         'proposed_headroom_usd', 'projected_upper_usd', 'cap_usd', 'spend_policy')}), flush=True)


def reserve(d, kind, value):
    headroom = float(value.get('max_configured_runtime_usd', IMAGE_BUILD_HEADROOM_USD))
    billing_snapshot(d, headroom)
    value.update(reserved_usd=0.0, budget_headroom_usd=headroom, requested_at=now(), state='creating')
    d[kind].append(value)
    save(d)

def latest(d):
    if not d['sessions'] or not d['sessions'][-1].get('sandbox_id'):
        raise RuntimeError('No confirmed Sandbox ID; recover any unknown creation before another allocation')
    return d['sessions'][-1], modal.Sandbox.from_id(d['sessions'][-1]['sandbox_id'])

def _mine(entry):
    # Entries from before workspaces were recorded belong to the original account.
    return entry.get('workspace', 'trxphxle') == WORKSPACE

def _image_id(d):
    # Newest snapshot taken in this workspace, else this workspace's base image.
    for entry in reversed(d['sessions']):
        if _mine(entry) and entry.get('snapshot_image_id'):
            return entry['snapshot_image_id']
    ids = d.get('image_ids', {})
    if WORKSPACE in ids:
        return ids[WORKSPACE]
    return d.get('image_id') if WORKSPACE == 'trxphxle' else None

def start():
    d = read()
    if d['sessions'] and _mine(d['sessions'][-1]):
        entry, sb = latest(d)
        status = sb.poll()
        if status is None:
            print(json.dumps(dict(reused=True, sandbox_id=sb.object_id, state='running')))
            return
        entry.update(state='finished', exit_code=status)
        entry.setdefault('observed_finished_at', now())
        save(d)
    if hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() != SOURCE_HASH:
        raise RuntimeError('Candidate hash differs from the independently reviewed source')
    app = modal.App.lookup(APP, create_if_missing=True)
    if _image_id(d):
        image = modal.Image.from_id(_image_id(d))
    else:
        if any(_mine(b) and not b.get('image_id') for b in d['image_builds']):
            raise RuntimeError('An image build already started without a confirmed ID; inspect/recover it first')
        entry = dict(image_reference=IMAGE, workspace=WORKSPACE)
        reserve(d, 'image_builds', entry)
        image = (modal.Image.from_registry(IMAGE).entrypoint([])
                 .add_local_file(CANDIDATE, '/workspace/candidate.py', copy=True)
                 .add_local_file(PROBE, '/workspace/resource_probe.py', copy=True))
        with modal.enable_output():
            image.build(app)
        entry.update(state='built', completed_at=now(), image_id=image.object_id)
        d.setdefault('image_ids', {})[WORKSPACE] = image.object_id
        d['image_id'] = image.object_id
        save(d)
    lifetime = allocation_seconds()
    name = f'okc-h200-{LANE + "-" if LANE else ""}{len(d["sessions"])+1:02d}'
    entry = dict(name=name, workspace=WORKSPACE, gpu='H200', timeout_seconds=lifetime, idle_timeout_seconds=min(300, lifetime),
                 cpu_limit=CPU, memory_limit_mib=MEMORY,
                 max_configured_runtime_usd=lifetime*UPPER_RATE)
    reserve(d, 'sessions', entry)
    # The server-side lifetime bounds charges even if this local process exits.
    sb = modal.Sandbox.create('/bin/sleep', str(lifetime), app=app, name=name, image=image, gpu='H200',
        timeout=lifetime, idle_timeout=min(300, lifetime), cpu=(CPU, CPU), memory=(MEMORY, MEMORY),
        block_network=True, workdir='/workspace',
        env={'MAX_JOBS':'4', 'TORCH_EXTENSIONS_DIR':'/tmp/okc-torch-extensions'},
        tags={'purpose':'openkernel-challenge', 'task':'01a18750'})
    entry.update(state='created', sandbox_id=sb.object_id, confirmed_at=now(),
                 confirmed_epoch=time.time())
    save(d)
    try:
        p = sb.exec('/opt/sglang/bin/python', '-c',
            'import torch,json; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability())))',
            timeout=180, stderr=modal.stream_type.StreamType.STDOUT)
        startup = p.stdout.read()
        p.wait()
        print(startup, flush=True)
        if p.returncode:
            raise RuntimeError(f'Modal readiness command exited with code {p.returncode}')
        entry.update(state='ready', readiness_at=now(), readiness_output=startup.strip())
        save(d)
    except BaseException as exc:
        entry.update(state='startup_failed', error=str(exc), last_checked_at=now())
        try:
            entry['exit_code'] = sb.terminate(wait=True)
        finally:
            entry['observed_finished_at'] = now()
            save(d)
        raise
    print(json.dumps(entry), flush=True)

def status(stop=False):
    d = read()
    entry, sb = latest(d)
    code = sb.terminate(wait=True) if stop else sb.poll()
    entry.update(state='running' if code is None else 'finished', exit_code=code,
                 last_checked_at=now())
    if code is not None:
        entry['observed_finished_at'] = now()
    save(d)
    print(json.dumps(entry), flush=True)

def run_probe():
    d = read()
    entry, sb = latest(d)
    if sb.poll() is not None:
        raise RuntimeError('Sandbox is no longer running')
    # Only task source files were baked into the image. No host config is mounted.
    with modal.enable_output():
        p = sb.exec('/opt/sglang/bin/python', '/workspace/resource_probe.py',
                    '/workspace/candidate.py', timeout=900,
                    stdout=modal.stream_type.StreamType.PIPE,
                    stderr=modal.stream_type.StreamType.STDOUT)
        for line in p.stdout:
            print(line, end='', flush=True)
        p.wait()
    entry.update(last_probe_returncode=p.returncode, last_probe_finished_at=now())
    save(d)
    if p.returncode:
        raise RuntimeError(f'Resource probe exited with code {p.returncode}')

def logs():
    d = read()
    entry, sb = latest(d)
    print('Sandbox', entry['sandbox_id'], 'exit', sb.poll(), flush=True)
    print('Dashboard:', sb.get_dashboard_url(), flush=True)
    print('STDOUT:', sb.stdout.read(), flush=True)
    print('STDERR:', sb.stderr.read(), flush=True)

def start_compare():
    # Validate the complete upload set before reserving or launching resources.
    for path, expected in ((CANDIDATE, SOURCE_HASH), (COMPARE, COMPARE_HASH), (NVFP4, NVFP4_HASH)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Reviewed source hash mismatch: {path.name}')
    start()
    d = read()
    entry, sb = latest(d)
    directory = OUT / 'modal-lab'
    directory.mkdir(exist_ok=True)
    stem = f'synthetic-compare-{entry["name"]}'
    local_result = directory / (stem + '.json')
    local_log = directory / (stem + '.log')
    remote_result = '/workspace/synthetic-two-wg.json'
    entry.update(comparison_started_at=now(), comparison_log=str(local_log),
                 comparison_result=str(local_result))
    save(d)
    try:
        sb.filesystem.write_bytes(COMPARE.read_bytes(), '/workspace/synthetic_compare.py')
        sb.filesystem.write_bytes(NVFP4.read_bytes(), '/workspace/nvfp4.py')
        p = sb.exec('/opt/sglang/bin/python', '/workspace/synthetic_compare.py',
            '/workspace/candidate.py', '/workspace/nvfp4.py', '--output', remote_result,
            '--pairs', '8', timeout=900, stdout=modal.stream_type.StreamType.PIPE,
            stderr=modal.stream_type.StreamType.STDOUT)
        with local_log.open('w') as log:
            for line in p.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
        p.wait()
        entry.update(comparison_returncode=p.returncode, comparison_finished_at=now())
        save(d)
        if p.returncode:
            raise RuntimeError(f'Synthetic comparison exited with code {p.returncode}')
    finally:
        try:
            result_text = sb.filesystem.read_text(remote_result)
            report = json.loads(result_text)
            if report.get('official_score') is not False:
                raise RuntimeError('Diagnostic result does not declare unofficial status')
            local_result.write_text(result_text)
            entry['comparison_result_status'] = report.get('status')
            print(json.dumps(dict(saved_result=str(local_result), status=report.get('status'))), flush=True)
            # Preserve successful compile artifacts for the next bounded run.
            if report.get('status') == 'passed_synthetic_comparisons':
                try:
                    snapshot = sb.snapshot_filesystem(timeout=55, ttl=86400)
                    entry['snapshot_image_id'] = snapshot.object_id
                    d['image_id'] = snapshot.object_id
                    d['image_expires_after'] = '24 hours from ' + now()
                except Exception as exc:
                    entry['snapshot_error'] = str(exc)
        except Exception as exc:
            entry['result_retrieval_error'] = str(exc)
            print(json.dumps(dict(result_retrieval_error=str(exc))), flush=True)
        finally:
            try:
                entry.update(exit_code=sb.terminate(wait=True), state='finished',
                             observed_finished_at=now())
            finally:
                save(d)
                print(json.dumps(dict(sandbox_id=sb.object_id, state=entry['state'],
                                      reserved_usd=d['reserved_usd'], actual_billed_usd='unknown')), flush=True)

def start_profile():
    uploads = ((COUNTER, COUNTER_HASH), (NCU_RUNNER, NCU_RUNNER_HASH),
               (COMPARE, COMPARE_HASH), (NVFP4, NVFP4_HASH))
    if COUNTER_HASH == 'REVIEW_PENDING' or NCU_FILTER == 'REVIEW_PENDING':
        raise RuntimeError('Counter driver is not yet independently reviewed')
    for path, expected in uploads:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Reviewed source hash mismatch: {path.name}')
    start()
    d = read()
    entry, sb = latest(d)
    directory = OUT / 'modal-lab' / ('counter-' + entry['name'])
    directory.mkdir(parents=True, exist_ok=True)
    entry.update(counter_started_at=now(), counter_output=str(directory))
    save(d)
    try:
        for path, _ in uploads:
            sb.filesystem.write_bytes(path.read_bytes(), '/workspace/' + path.name)
        p = sb.exec('/opt/sglang/bin/python', '/workspace/ncu_runner.py',
                    '--kernel-filter', NCU_FILTER, timeout=900,
                    stdout=modal.stream_type.StreamType.PIPE,
                    stderr=modal.stream_type.StreamType.STDOUT)
        with (directory / 'runner.log').open('w') as log:
            for line in p.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
        p.wait()
        entry.update(counter_returncode=p.returncode, counter_finished_at=now())
        save(d)
        if p.returncode:
            raise RuntimeError(f'Counter diagnostic exited with code {p.returncode}')
    finally:
        # Retrieve only the known diagnostic outputs, never host/config files.
        try:
            missing = {}
            for name in ('status.json', 'smoke.log', 'smoke.csv', 'metrics.txt',
                         'profile.log', 'profile.csv', 'counter-probe.json'):
                try:
                    contents = sb.filesystem.read_text('/workspace/ncu-diagnostic/' + name)
                    (directory / name).write_text(contents)
                except Exception as exc:
                    missing[name] = str(exc)
            entry['counter_unavailable_outputs'] = missing
        finally:
            try:
                entry.update(exit_code=sb.terminate(wait=True), state='finished',
                             observed_finished_at=now())
            finally:
                save(d)
                print(json.dumps(dict(sandbox_id=sb.object_id, state=entry['state'],
                                      saved_directory=str(directory), reserved_usd=d['reserved_usd'],
                                      actual_billed_usd='unknown')), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start','start-probe','start-compare','start-profile','status','probe','stop','logs','budget'))
    args = parser.parse_args()
    if args.action == 'start-probe':
        start()
        run_probe()
    else:
        {'start':start, 'start-compare':start_compare, 'start-profile':start_profile, 'status':status, 'probe':run_probe, 'stop':lambda:status(True), 'logs':logs, 'budget':budget}[args.action]()
