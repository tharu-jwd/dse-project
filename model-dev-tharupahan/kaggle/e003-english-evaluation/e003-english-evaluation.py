"""Evaluate E003 on the frozen LibriSpeech test-clean benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "e003-english-evaluation"
MODEL = WORK / "whisper-small"
ADAPTER = WORK / "e003-adapter"


def one_parent(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0].parent


def main() -> None:
    runtime = one_parent("whisper-small--model.safetensors")
    evaluation = one_parent("english-retention-test-clean.parquet")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(runtime),
            "transformers==5.16.1",
            "peft==0.20.0",
            "huggingface-hub==1.30.0",
            "tokenizers==0.23.2",
            "safetensors==0.8.0",
            "accelerate==1.14.0",
        ],
        check=True,
    )
    MODEL.mkdir()
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    ADAPTER.mkdir()
    for asset in evaluation.glob("adapter--*"):
        (ADAPTER / asset.name.removeprefix("adapter--")).symlink_to(asset)
    if not (ADAPTER / "adapter_model.safetensors").is_file():
        raise RuntimeError("E003 adapter is incomplete")
    environment = os.environ.copy()
    environment["SINHALA_ASR_MODEL"] = str(MODEL)
    subprocess.run(
        [
            sys.executable,
            str(evaluation / "run_english_retention.py"),
            "--benchmark",
            str(evaluation / "english-retention-test-clean.parquet"),
            "--output-dir",
            str(OUTPUT),
            "--adapter",
            str(ADAPTER),
            "--variant",
            "e003",
        ],
        check=True,
        env=environment,
    )
    runtime_metadata = json.loads((OUTPUT / "e003-runtime.json").read_text())
    result = {
        "state": "complete",
        "rows": runtime_metadata["rows"],
        "predictions_sha256": runtime_metadata["predictions_sha256"],
        "benchmark_sha256": runtime_metadata["benchmark_sha256"],
    }
    (WORK / "e003-kaggle-english-evaluation-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
