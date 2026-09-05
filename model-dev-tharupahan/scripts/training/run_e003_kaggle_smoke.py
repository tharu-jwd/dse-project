#!/usr/bin/env python3
"""Run the gated two-phase E003 checkpoint-resume smoke on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
OUTPUT = WORK / "e003-smoke"
RUNTIME_ASSETS = Path(__file__).resolve().parent
RUNTIME_INPUT = WORK / "e003-input"
MODEL = WORK / "whisper-small"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "SINHALA_ASR_JOB_ROOT": str(WORK / "runtime"),
            "SINHALA_ASR_INPUT_ROOT": str(RUNTIME_INPUT),
            "SINHALA_ASR_OUTPUT_ROOT": str(OUTPUT),
            "SINHALA_ASR_JOB_CONFIG": str(config),
            "SINHALA_ASR_MODEL": str(MODEL),
        }
    )
    subprocess.run(
        [sys.executable, str(RUNTIME_ASSETS / "run_e002_colab.py")],
        check=True,
        env=environment,
    )


def main() -> None:
    manifests = list(Path("/kaggle/input").rglob("train-manifest.json"))
    if len(manifests) != 1:
        raise SystemExit(f"expected one E003 train manifest, found {manifests}")
    source_input = manifests[0].parent
    shutil.rmtree(OUTPUT, ignore_errors=True)
    shutil.rmtree(RUNTIME_INPUT, ignore_errors=True)
    shutil.rmtree(MODEL, ignore_errors=True)
    MODEL.mkdir()
    for source in RUNTIME_ASSETS.glob("whisper-small--*"):
        (MODEL / source.name.removeprefix("whisper-small--")).symlink_to(source)
    if not (MODEL / "model.safetensors").is_file():
        raise SystemExit("offline Whisper-small model is incomplete")
    (RUNTIME_INPUT / "train").mkdir(parents=True)
    manifest = source_input / "train-manifest.json"
    expected_manifest = RUNTIME_INPUT / "train" / "manifest.json"
    expected_manifest.symlink_to(manifest)
    for shard in json.loads(manifest.read_text())["shards"]:
        (expected_manifest.parent / shard["name"]).symlink_to(
            source_input / shard["name"]
        )
    (RUNTIME_INPUT / "v4-validation-206-audio.parquet").symlink_to(
        source_input / "v4-validation-206-audio.parquet"
    )
    phase_a = source_input / "e003-smoke-phase-a-job-config.json"
    run(phase_a)
    checkpoint = OUTPUT / "trainer/checkpoint-1"
    if not (checkpoint / "COMPLETE").is_file():
        raise SystemExit("phase A did not produce complete checkpoint 1")
    phase_b_config = json.loads(
        (source_input / "e003-smoke-phase-b-job-config.json").read_text()
    )
    phase_b_config["resume_from_checkpoint"] = str(checkpoint)
    phase_b = WORK / "e003-smoke-phase-b-resolved.json"
    phase_b.write_text(json.dumps(phase_b_config, indent=2) + "\n")
    run(phase_b)
    checkpoint_two = OUTPUT / "trainer/checkpoint-2"
    if not (checkpoint_two / "COMPLETE").is_file():
        raise SystemExit("phase B did not produce complete checkpoint 2")
    index = json.loads((OUTPUT / "checkpoint-index.json").read_text())
    result = {
        "state": "complete",
        "phase_a_checkpoint": index[0] if isinstance(index, list) else index["checkpoints"][0],
        "phase_b_checkpoint": index[-1] if isinstance(index, list) else index["checkpoints"][-1],
        "phase_b_resolved_sha256": sha256(phase_b),
    }
    (WORK / "e003-kaggle-smoke-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
