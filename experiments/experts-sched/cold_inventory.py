"""Read-only installed-NCU inventory for the cold-start diagnostic."""
import json
from pathlib import Path
import re
import subprocess
import torch

ROOT = Path('/workspace')
NCU = '/usr/local/cuda/bin/ncu'
report = {'status': 'running', 'artifact_files': [], 'gpu': torch.cuda.get_device_name(),
          'torch': torch.__version__, 'cuda': torch.version.cuda, 'commands': []}
output = ROOT / 'cold-inventory.json'
try:
    for name, arguments in (('version', ['--version']), ('help', ['--help']),
                            ('metrics', ['--query-metrics', '--query-metrics-mode', 'all'])):
        command = [NCU, *arguments]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=90, check=False)
        filename = 'cold-ncu-' + name + '.txt'
        (ROOT / filename).write_text(result.stdout)
        report['artifact_files'].append(filename)
        report['commands'].append({'command': command, 'returncode': result.returncode})
        if result.returncode:
            raise RuntimeError('NCU inventory command failed: ' + name)
        if name == 'metrics':
            report['instruction_metric_lines'] = [line for line in result.stdout.splitlines()
                if re.search(r'no_instruction|icc_|icache|instruction_cache|imc_miss|instruction_fetch', line, re.I)]
        if name == 'help':
            lines = result.stdout.splitlines()
            selected = set()
            for index, line in enumerate(lines):
                if any(term in line for term in ('--replay-mode', '--cache-control', '--graph-profiling', '--nvtx-include')):
                    selected.update(range(index, min(len(lines), index+18)))
            report['replay_help'] = [lines[index] for index in sorted(selected)]
    report['status'] = 'complete'
except BaseException as exc:
    report['status'] = 'failed'
    report['error'] = repr(exc)
    raise
finally:
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'instruction_metric_lines': report.get('instruction_metric_lines', [])}), flush=True)
