"""Kaggle entry point for the gated E003 resume smoke."""

import subprocess
import sys
from pathlib import Path

input_root = Path("/kaggle/input/sinhala-asr-e003-inputs")
runtime_root = Path("/kaggle/input/sinhala-asr-e003-runtime")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--find-links",
        str(runtime_root),
        "transformers==5.16.1",
        "peft==0.20.0",
        "huggingface-hub==1.30.0",
        "tokenizers==0.23.2",
        "safetensors==0.8.0",
        "accelerate==1.14.0",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, str(runtime_root / "run_e003_kaggle_smoke.py")], check=True
)
