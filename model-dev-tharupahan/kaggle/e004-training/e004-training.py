"""Run E004 with untouched-Whisper teacher targets for English replay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "e004-training"
LOCAL_INPUT = WORK / "e004-input"
MODEL = WORK / "whisper-small"
E003_MANIFEST_SHA256 = "51e69bfa00da0f5f0d4b89c0c31dfb2d28f6e39766f3d23b74e8e1e23f86190d"
E003_CONFIG_SHA256 = "ff7e04195279b86f9809a9becb86539e6766b9c97e1b0ef4faaf9f24604db0a1"
TEACHER_SHA256 = "5f45c712867f430ab2e911cfa8950634c69b5897689b082486c8c6f1fc45c0f6"
E004_FINGERPRINT = "d6e29790c9cd17f26055c98a92ce784edacb8599635d26ce3d177dc0cc816899"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_parent(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0].parent


def install_runtime(runtime: Path) -> None:
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


def stage_inputs(runtime: Path, source: Path, teacher_path: Path) -> Path:
    if sha256(source / "train-manifest.json") != E003_MANIFEST_SHA256:
        raise RuntimeError("E003 manifest transport hash mismatch")
    if sha256(source / "e003-job-config.json") != E003_CONFIG_SHA256:
        raise RuntimeError("E003 config transport hash mismatch")
    if sha256(teacher_path) != TEACHER_SHA256:
        raise RuntimeError("teacher-label transport hash mismatch")
    teacher_rows = pq.read_table(teacher_path).to_pylist()
    teachers = {row["sample_id"]: row for row in teacher_rows}
    if len(teachers) != 1111 or len(teacher_rows) != 1111:
        raise RuntimeError("teacher-label identity mismatch")

    shutil.rmtree(LOCAL_INPUT, ignore_errors=True)
    shutil.rmtree(MODEL, ignore_errors=True)
    MODEL.mkdir()
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    if not (MODEL / "model.safetensors").is_file():
        raise RuntimeError("offline Whisper-small model is incomplete")

    train = LOCAL_INPUT / "train"
    train.mkdir(parents=True)
    source_manifest = json.loads((source / "train-manifest.json").read_text())
    staged_shards = []
    seen_teacher_ids: set[str] = set()
    for shard in source_manifest["shards"]:
        source_shard = source / shard["name"]
        if sha256(source_shard) != shard["sha256"]:
            raise RuntimeError(f"source shard hash mismatch: {shard['name']}")
        destination = train / shard["name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source_shard)
        rows = table.to_pylist()
        english = [row for row in rows if (row.get("decoder_language") or "si") == "en"]
        if english:
            if len(english) != len(rows):
                raise RuntimeError(f"mixed-language source shard: {shard['name']}")
            replaced = []
            for row in rows:
                teacher = teachers.get(row["sample_id"])
                if teacher is None:
                    raise RuntimeError(f"missing teacher label: {row['sample_id']}")
                if teacher["audio_sha256"] != row["audio_sha256"] or teacher["text"] != row["text"]:
                    raise RuntimeError(f"teacher/source mismatch: {row['sample_id']}")
                replaced.append({**row, "text": teacher["teacher_text"]})
                seen_teacher_ids.add(row["sample_id"])
            pq.write_table(pa.Table.from_pylist(replaced, schema=table.schema), destination, compression="zstd")
        else:
            destination.symlink_to(source_shard)
        staged_shards.append(
            {
                "name": shard["name"],
                "rows": len(rows),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    if seen_teacher_ids != set(teachers):
        raise RuntimeError("not every teacher label was applied exactly once")
    manifest = {
        **source_manifest,
        "experiment": "e004",
        "source_sha256": E004_FINGERPRINT,
        "teacher_labels_sha256": TEACHER_SHA256,
        "shards": staged_shards,
    }
    (train / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (LOCAL_INPUT / "v4-validation-206-audio.parquet").symlink_to(
        source / "v4-validation-206-audio.parquet"
    )
    config = json.loads((source / "e003-job-config.json").read_text())
    config.update(
        {
            "experiment": "e004",
            "output_dir": "runs/e004-whisper-small-wide-lora-r16-teacher-replay-v4",
            "training_bundle_sha256": E004_FINGERPRINT,
            "teacher_labels_sha256": TEACHER_SHA256,
            "git_commit": "bfd1055",
            "tracked_config": "configs/training/experiments/e004-wide-lora-r16-500-step-teacher-replay-v4.json",
        }
    )
    config_path = LOCAL_INPUT / "e004-job-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def main() -> None:
    runtime = one_parent("whisper-small--model.safetensors")
    source = one_parent("train-manifest.json")
    teacher_path = next(iter(INPUTS.rglob("e004-english-teacher-labels.parquet")), None)
    if teacher_path is None:
        raise RuntimeError("E004 teacher labels are not attached")
    install_runtime(runtime)
    config = stage_inputs(runtime, source, teacher_path)
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
        [sys.executable, str(runtime / "run_e002_colab.py")], check=True, env=environment
    )
    status = json.loads((OUTPUT / "status.json").read_text())
    metadata = json.loads((OUTPUT / "run-metadata.json").read_text())
    checkpoints = json.loads((OUTPUT / "checkpoint-index.json").read_text())
    if status != {"state": "complete", "exit_code": 0}:
        raise RuntimeError(f"unexpected final status: {status}")
    if checkpoints["checkpoints"][-1]["step"] != 500:
        raise RuntimeError("final durable checkpoint is not step 500")
    if float(metadata["train_metrics"]["train_steps_per_second"]) <= 0:
        raise RuntimeError("training throughput was not positive")
    predictions = OUTPUT / "sinhala-validation-predictions.parquet"
    result = {
        "state": "complete",
        "steps": 500,
        "training_fingerprint": E004_FINGERPRINT,
        "predictions_sha256": sha256(predictions),
        "adapter_sha256": sha256(OUTPUT / "final-adapter/adapter_model.safetensors"),
        "checkpoint_index_sha256": sha256(OUTPUT / "checkpoint-index.json"),
    }
    (WORK / "e004-kaggle-training-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
