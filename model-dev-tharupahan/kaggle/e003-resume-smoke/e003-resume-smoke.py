"""Kaggle entry point for the gated E003 resume smoke."""

import subprocess
import sys
from pathlib import Path

input_root = Path("/kaggle/input/sinhala-asr-e003-inputs")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "transformers==5.16.1",
        "peft==0.20.0",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, str(input_root / "run_e003_kaggle_smoke.py")], check=True
)
