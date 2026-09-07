#!/usr/bin/env python3
"""E013: score E007 against dataset v5's new test set and a validation slice.

Two runs of scripts/evaluation/predict.py (beam5, the project's own
recommended default -- see plan.md item 1), both against E007's real,
hash-verified adapter:

1. Full test set (10,423 rows, 22.72h -- Path Nirvana TTS + SPEAK-ASR
   YouTube, zero OpenSLR). E007 has never seen this data in training or in
   any prior evaluation -- a genuine generalization check.
2. A 200-row deterministic slice of v5's openslr-validation-v2 (52 held-out
   speakers). NOT the full 24.98h/20,232-row validation set -- packaging
   that at full scale needs resolving ~25h of individual OpenSLR FLAC files
   into a self-contained transport, deferred as a separate task. This slice
   reuses the same 200 rows already packaged for E012's search pilot.

Scoring (strict/canonical WER/CER, by-source breakdown) happens locally
after download, not in this kernel -- keeps the kernel to what needs the
GPU.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
MODEL = WORK / "whisper-small"
ADAPTER_SHA256 = "4c92dee2fbdc7bc59ef70e7e17654967b47cf8b535956f67a09c2c1458acc865"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_file(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0]


def one_dir(name: str) -> Path:
    matches = [p for p in INPUTS.rglob(name) if p.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one dir {name}, found {matches}")
    return matches[0]


def install_runtime(runtime: Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
            "--find-links", str(runtime),
            "transformers==5.16.1", "peft==0.20.0", "huggingface-hub==1.30.0",
            "tokenizers==0.23.2", "safetensors==0.8.0", "accelerate==1.14.0",
        ],
        check=True,
    )


def stage_model(runtime: Path) -> None:
    shutil.rmtree(MODEL, ignore_errors=True)
    MODEL.mkdir(parents=True)
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    if not (MODEL / "model.safetensors").is_file():
        raise RuntimeError("offline Whisper-small model is incomplete")


def main() -> None:
    runtime = one_file("whisper-small--model.safetensors").parent
    predict_script = one_file("predict.py")
    sinhala_asr_src = one_dir("sinhala_asr")
    adapter_dir = one_file("adapter_model.safetensors").parent

    adapter_hash = sha256(adapter_dir / "adapter_model.safetensors")
    if adapter_hash != ADAPTER_SHA256:
        raise RuntimeError(f"E007 adapter hash mismatch: {adapter_hash} != {ADAPTER_SHA256}")

    install_runtime(runtime)
    stage_model(runtime)

    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = str(sinhala_asr_src.parent)

    test_manifest = one_file("manifest.parquet")  # external-eval-v1's own manifest
    # Disambiguate from e012-search-inputs' manifest-replay*.parquet, which
    # also contain a `dataset_split == "validation"` slice (the same 200
    # rows for all three, reused here rather than repackaged).
    validation_manifest = one_file("manifest-replay10.parquet")

    runs = [
        ("test", test_manifest, True, WORK / "e013-test-predictions.parquet"),
        ("validation", validation_manifest, False, WORK / "e013-validation-predictions.parquet"),
    ]
    for split, manifest, unlock_test, output in runs:
        print(f"=== scoring {split} ===", flush=True)
        args = [
            sys.executable, str(predict_script),
            "--model", str(MODEL),
            "--adapter", str(adapter_dir),
            "--manifest", str(manifest),
            "--split", split,
            "--output", str(output),
            "--batch-size", "8",
            "--num-beams", "5",
        ]
        if unlock_test:
            args.append("--unlock-test")
        subprocess.run(args, check=True, env=subprocess_env)


if __name__ == "__main__":
    main()
