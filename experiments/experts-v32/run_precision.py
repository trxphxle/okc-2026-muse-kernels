"""Lane-local launcher: inherit the active workspace's prebuilt MAIN image.

Shared lab.py is read-only. Existing lane ledgers predate image_ids, so its
normal read() early-return otherwise misses MAIN's newly built account image.
Uses the active default Modal profile; never overrides MODAL_PROFILE.
"""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "work/modal"))
import lab

_original_read = lab.read


def read_with_workspace_image():
    ledger = _original_read()
    if lab.WORKSPACE not in ledger.get("image_ids", {}):
        main = json.loads(lab.MAIN_LEDGER.read_text())
        image = main.get("image_ids", {}).get(lab.WORKSPACE)
        if image:
            ledger.setdefault("image_ids", {})[lab.WORKSPACE] = image
    return ledger


lab.read = read_with_workspace_image
runpy.run_path(str(ROOT / "work/modal/run_layer.py"), run_name="__main__")
