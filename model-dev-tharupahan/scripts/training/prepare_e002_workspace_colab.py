#!/usr/bin/env python3
"""Create an isolated E002 workspace and validate the disposable runtime."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

root = Path("/content/sinhala-asr-job")
if root.exists() and any(root.iterdir()):
    raise SystemExit(f"refusing to reuse non-empty workspace: {root}")
(root / "input" / "train").mkdir(parents=True, exist_ok=True)
(root / "output").mkdir(parents=True, exist_ok=True)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True
)
disk = shutil.disk_usage(root)
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU unavailable")
metadata = {
    "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.version.cuda,
    "disk_total_bytes": disk.total,
    "disk_free_bytes": disk.free,
    "python": sys.version,
}
(root / "output" / "session.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(metadata, indent=2))
