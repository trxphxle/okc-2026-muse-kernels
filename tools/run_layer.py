"""Run an explicit local diagnostic in the bounded warm H200 sandbox.

Root only. Reuses lab.py's budget ledger, lifetime and one-GPU policy. Successful
runs preserve compiled artifacts; failures terminate the sandbox. No deployment.
"""
import argparse
import json
from pathlib import Path
import modal
import lab

parser = argparse.ArgumentParser()
parser.add_argument('script', type=Path)
parser.add_argument('files', nargs='*', type=Path)
parser.add_argument('--result', default='layer-result.json')
parser.add_argument('--args', dest='script_args', nargs=argparse.REMAINDER, default=[])
args = parser.parse_args()
paths = [args.script, *args.files]
if len({p.name for p in paths}) != len(paths):
    raise ValueError('Upload basenames must be unique')
contents = [(p.name, p.read_bytes()) for p in paths]
lab.start()
ledger = lab.read()
entry, sandbox = lab.latest(ledger)
directory = lab.OUT / 'modal-lab' / ('layer-' + entry['name'])
directory.mkdir(exist_ok=True)
entry['last_layer_exit_code'] = None
try:
    for name, data in contents:
        sandbox.filesystem.write_bytes(data, '/workspace/' + name)
    proc = sandbox.exec('/opt/sglang/bin/python', '/workspace/' + args.script.name, *args.script_args,
        timeout=900, stdout=modal.stream_type.StreamType.PIPE,
        stderr=modal.stream_type.StreamType.STDOUT)
    with (directory / (args.script.stem + '.log')).open('w') as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
    proc.wait()
    entry['last_layer_exit_code'] = proc.returncode
    if proc.returncode:
        raise RuntimeError(f'Diagnostic exited {proc.returncode}')
finally:
    try:
        text = sandbox.filesystem.read_text('/workspace/' + args.result)
        (directory / args.result).write_text(text)
        print('Saved ' + str(directory / args.result), flush=True)
        for name in json.loads(text).get('artifact_files', []):
            if Path(name).name != name:
                raise ValueError('Diagnostic artifact must be a basename')
            (directory / name).write_bytes(sandbox.filesystem.read_bytes('/workspace/' + name))
            print('Saved ' + str(directory / name), flush=True)
    except Exception as exc:
        entry['last_layer_retrieval_error'] = str(exc)
    try:
        # Preserve code/build caches even when a numerical diagnostic rejects
        # a candidate. The next process still revalidates its actual outputs.
        snapshot = sandbox.snapshot_filesystem(timeout=55, ttl=86400)
        entry['snapshot_image_id'] = snapshot.object_id
        ledger['image_id'] = snapshot.object_id
        ledger['image_expires_after'] = '24 hours from ' + lab.now()
    except Exception as exc:
        entry['snapshot_error'] = str(exc)
    if entry.get('last_layer_exit_code') != 0:
        entry.update(exit_code=sandbox.terminate(wait=True), state='finished',
                     observed_finished_at=lab.now())
    entry['last_layer_script'] = args.script.name
    entry['last_layer_finished_at'] = lab.now()
    lab.save(ledger)
