"""Run the unchanged shared Modal tools with lane-local SDK and recovery ledger.

The first new-workspace build failed in the old SDK, leaving an unresolved build
reservation in the shared lane ledger. That ledger is preserved read-only. The
new ledger retains its history and the failed reservation separately; no old
account resource is accessed. Active profile is always Modal's default.
"""
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / '.modal-sdk'))
sys.path.insert(0, str(ROOT / 'work/modal'))
import lab

if os.environ.get('OKC_LANE') != 'attn-overlap':
    raise RuntimeError('Set OKC_LANE=attn-overlap')
if lab.WORKSPACE != 'secondary-workspace':
    raise RuntimeError('Default Modal workspace changed; inspect before allocation')
lab.LEDGER = HERE / 'modal-budget-workspace.json'
if len(sys.argv) > 1 and sys.argv[1] == '--lab':
    operations = {'stop': lambda: lab.status(True), 'status': lab.status, 'budget': lab.budget}
    if len(sys.argv) != 3 or sys.argv[2] not in operations:
        raise SystemExit('Usage: modal_lane.py --lab stop|status|budget')
    operations[sys.argv[2]]()
else:
    runpy.run_path(str(ROOT / 'work/modal/run_layer.py'), run_name='__main__')
