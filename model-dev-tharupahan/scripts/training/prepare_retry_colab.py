#!/usr/bin/env python3
"""Archive one failed remote attempt and remove incompatible optional torchao."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

root = Path("/content/sinhala-asr-job")
output = root / "output"
archive = root / "failed-attempt-001"
if archive.exists():
    raise SystemExit(f"refusing to overwrite {archive}")
if output.exists():
    output.rename(archive)
output.mkdir(parents=True)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True
)
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=True)
print("retry environment prepared")
