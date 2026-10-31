"""Complete this lane's image build after the old SDK rejected the builder version.

Uses the default active profile and shared lab bookkeeping; never overrides the
profile, removes another lane's entry, or invents an image ID.
"""
import os
from pathlib import Path
import sys

if os.environ.get('OKC_LANE') != 'experts-sched':
    raise RuntimeError('Explicit experts-sched lane required')
root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'work' / 'modal'))
import modal
import lab

ledger = lab.read()
existing = lab._image_id(ledger)
if existing:
    print({'status': 'already_available', 'workspace': lab.WORKSPACE, 'image_id': existing}, flush=True)
else:
    pending = [entry for entry in ledger['image_builds'] if lab._mine(entry) and not entry.get('image_id')]
    if len(pending) != 1:
        raise RuntimeError('Expected exactly one known pending lane image build')
    requested_at = pending[0]['requested_at']
    app = modal.App.lookup(lab.APP, create_if_missing=True)
    image = (modal.Image.from_registry(lab.IMAGE).entrypoint([])
             .add_local_file(lab.CANDIDATE, '/workspace/candidate.py', copy=True)
             .add_local_file(lab.PROBE, '/workspace/resource_probe.py', copy=True))
    with modal.enable_output():
        image.build(app)
    print({'status': 'built', 'workspace': lab.WORKSPACE, 'image_id': image.object_id}, flush=True)
    # Re-read after the asynchronous service operation to retain intervening
    # bookkeeping changes; identify only the original failed reservation.
    ledger = lab.read()
    matches = [entry for entry in ledger['image_builds']
               if lab._mine(entry) and entry.get('requested_at') == requested_at]
    if len(matches) != 1:
        raise RuntimeError('Pending reservation changed during image recovery')
    matches[0].update(state='built', completed_at=lab.now(), image_id=image.object_id,
                      recovery='Current SDK after unsupported-builder client failure')
    ledger.setdefault('image_ids', {})[lab.WORKSPACE] = image.object_id
    ledger['image_id'] = image.object_id
    lab.save(ledger)
