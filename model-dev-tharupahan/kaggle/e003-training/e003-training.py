"""Run the controlled 500-step E003 English-replay experiment on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "e003-training"
LOCAL_INPUT = WORK / "e003-input"
MODEL = WORK / "whisper-small"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_runtime() -> Path:
    candidates = [
        path.parent
        for path in INPUTS.rglob("whisper-small--model.safetensors")
        if (path.parent / "run_e002_colab.py").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one E003 runtime dataset, found {candidates}")
    return candidates[0]


def install_runtime(runtime: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        check=True,
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


def stage_inputs(runtime: Path) -> Path:
    manifests = list(INPUTS.rglob("train-manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected one E003 train manifest, found {manifests}")
    source = manifests[0].parent
    shutil.rmtree(LOCAL_INPUT, ignore_errors=True)
    shutil.rmtree(MODEL, ignore_errors=True)
    MODEL.mkdir()
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    if not (MODEL / "model.safetensors").is_file():
        raise RuntimeError("offline Whisper-small model is incomplete")
    train = LOCAL_INPUT / "train"
    train.mkdir(parents=True)
    manifest = source / "train-manifest.json"
    (train / "manifest.json").symlink_to(manifest)
    for shard in json.loads(manifest.read_text())["shards"]:
        (train / shard["name"]).symlink_to(source / shard["name"])
    (LOCAL_INPUT / "v4-validation-206-audio.parquet").symlink_to(
        source / "v4-validation-206-audio.parquet"
    )
    return source / "e003-job-config.json"


def main() -> None:
    runtime = find_runtime()
    install_runtime(runtime)
    config = stage_inputs(runtime)
    environment = os.environ.copy()
    environment.update(
        {
            "SINHALA_ASR_JOB_ROOT": str(WORK / "runtime"),
            "SINHALA_ASR_INPUT_ROOT": str(LOCAL_INPUT),
            "SINHALA_ASR_OUTPUT_ROOT": str(OUTPUT),
            "SINHALA_ASR_JOB_CONFIG": str(config),
            "SINHALA_ASR_MODEL": str(MODEL),
        }
    )
    subprocess.run(
        [sys.executable, str(runtime / "run_e002_colab.py")],
        check=True,
        env=environment,
    )
    status = json.loads((OUTPUT / "status.json").read_text())
    metadata = json.loads((OUTPUT / "run-metadata.json").read_text())
    checkpoint_index = json.loads((OUTPUT / "checkpoint-index.json").read_text())
    if status != {"state": "complete", "exit_code": 0}:
        raise RuntimeError(f"unexpected final status: {status}")
    if float(metadata["train_metrics"]["train_steps_per_second"]) <= 0:
        raise RuntimeError("training throughput was not positive")
    if checkpoint_index["checkpoints"][-1]["step"] != 500:
        raise RuntimeError("final durable checkpoint is not step 500")
    predictions = OUTPUT / "sinhala-validation-predictions.parquet"
    if not predictions.is_file():
        raise RuntimeError("canonical Sinhala validation predictions are missing")
    result = {
        "state": "complete",
        "steps": 500,
        "predictions_sha256": sha256(predictions),
        "adapter_sha256": sha256(OUTPUT / "final-adapter/adapter_model.safetensors"),
        "checkpoint_index_sha256": sha256(OUTPUT / "checkpoint-index.json"),
    }
    (WORK / "e003-kaggle-training-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
