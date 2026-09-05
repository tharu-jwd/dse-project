"""Scale E005 to 82,835 Sinhala rows (100 hours) while retaining teacher
behavior replay."""

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
OUTPUT = WORK / "e006-training"
LOCAL_INPUT = WORK / "e006-input"
MODEL = WORK / "whisper-small"
SINHALA_SHARDS_COMBINED_SHA256 = (
    "2f55ff4ea8bb7e4eebfc4cd781e5358b907def6c9a5cebd9a54fad8c2a6246e7"
)
ENGLISH_SHA256 = "0246f185b9f08a79e1eec91f7effb07e7912dd3199b00d78a72cfef61c54781b"
TEACHER_SHA256 = "5f45c712867f430ab2e911cfa8950634c69b5897689b082486c8c6f1fc45c0f6"
VALIDATION_SHA256 = "c7a378a115dd953c50bfe6a2c550e28b5a5cc829dee79bdb29ccde1151e46ec2"
E006_FINGERPRINT = "c156fc2be0dc7c579e3ad01406f0ed4ac9325ac597ca119d5c03d9d35c2c3c9b"
SINHALA_ROWS = 82_835
ENGLISH_UNIQUE_ROWS = 1_111
ENGLISH_FULL_COPIES = 8
ENGLISH_PARTIAL_ROWS = 316
ENGLISH_ROWS = ENGLISH_FULL_COPIES * ENGLISH_UNIQUE_ROWS + ENGLISH_PARTIAL_ROWS
TRAINING_ROWS = SINHALA_ROWS + ENGLISH_ROWS
MAX_STEPS = 2_870


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


def stage_model(runtime: Path) -> None:
    shutil.rmtree(MODEL, ignore_errors=True)
    MODEL.mkdir()
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    if not (MODEL / "model.safetensors").is_file():
        raise RuntimeError("offline Whisper-small model is incomplete")


def load_english(source: Path, teacher_path: Path) -> list[dict]:
    source_manifest_path = source / "train-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest["english_replay_source_sha256"] != ENGLISH_SHA256:
        raise RuntimeError("E003 English source fingerprint mismatch")
    if sha256(teacher_path) != TEACHER_SHA256:
        raise RuntimeError("teacher-label transport hash mismatch")
    teacher_rows = pq.read_table(teacher_path).to_pylist()
    teachers = {row["sample_id"]: row for row in teacher_rows}
    if len(teacher_rows) != ENGLISH_UNIQUE_ROWS or len(teachers) != ENGLISH_UNIQUE_ROWS:
        raise RuntimeError("teacher-label identity mismatch")
    english = []
    for shard in source_manifest["shards"]:
        path = source / shard["name"]
        if sha256(path) != shard["sha256"]:
            raise RuntimeError(f"E003 shard hash mismatch: {path.name}")
        rows = pq.read_table(path).to_pylist()
        languages = {row.get("decoder_language") or "si" for row in rows}
        if languages == {"en"}:
            english.extend(rows)
        elif "en" in languages:
            raise RuntimeError(f"mixed-language E003 shard: {path.name}")
    english.sort(key=lambda row: row["sample_id"])
    if len(english) != ENGLISH_UNIQUE_ROWS or {
        row["sample_id"] for row in english
    } != set(teachers):
        raise RuntimeError("E003 English rows do not match teacher IDs")
    replaced = []
    for row in english:
        teacher = teachers[row["sample_id"]]
        if teacher["audio_sha256"] != row["audio_sha256"] or teacher["text"] != row["text"]:
            raise RuntimeError(f"teacher/source mismatch: {row['sample_id']}")
        replaced.append({**row, "text": teacher["teacher_text"]})
    return replaced


def stage_inputs(sinhala: Path, english_source: Path, teacher_path: Path) -> Path:
    manifest_path = sinhala / "sinhala-manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest["shards_combined_sha256"] != SINHALA_SHARDS_COMBINED_SHA256:
        raise RuntimeError("E006 Sinhala source fingerprint mismatch")
    if source_manifest["rows"] != SINHALA_ROWS:
        raise RuntimeError("E006 Sinhala row count mismatch")

    shutil.rmtree(LOCAL_INPUT, ignore_errors=True)
    train = LOCAL_INPUT / "train"
    train.mkdir(parents=True)
    staged_shards = []
    total_sinhala = 0
    for shard in source_manifest["shards"]:
        source = sinhala / shard["name"]
        if sha256(source) != shard["sha256"]:
            raise RuntimeError(f"Sinhala shard hash mismatch: {source.name}")
        destination = train / shard["name"]
        destination.symlink_to(source)
        total_sinhala += int(shard["rows"])
        staged_shards.append(dict(shard))
    if total_sinhala != SINHALA_ROWS:
        raise RuntimeError("staged Sinhala row count mismatch")

    unique_english = load_english(english_source, teacher_path)
    replay = []
    for occurrence in range(ENGLISH_FULL_COPIES):
        for row in unique_english:
            replay.append(
                {
                    **row,
                    "sample_id": f"{row['sample_id']}#e006-replay-{occurrence}",
                }
            )
    for row in unique_english[:ENGLISH_PARTIAL_ROWS]:
        replay.append(
            {
                **row,
                "sample_id": f"{row['sample_id']}#e006-replay-{ENGLISH_FULL_COPIES}",
            }
        )
    if len(replay) != ENGLISH_ROWS or len({row["sample_id"] for row in replay}) != ENGLISH_ROWS:
        raise RuntimeError("English replay occurrence identity mismatch")
    for start in range(0, len(replay), 1_000):
        rows = replay[start : start + 1_000]
        name = f"english-replay-part-{start // 1000:04d}.parquet"
        destination = train / name
        pq.write_table(pa.Table.from_pylist(rows), destination, compression="zstd")
        staged_shards.append(
            {
                "name": name,
                "rows": len(rows),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )

    manifest = {
        "experiment": "e006",
        "source_sha256": E006_FINGERPRINT,
        "rows": TRAINING_ROWS,
        "sinhala_source_rows": SINHALA_ROWS,
        "english_replay_rows": ENGLISH_ROWS,
        "english_unique_rows": len(unique_english),
        "english_oversampling": (
            f"{ENGLISH_FULL_COPIES} full stable-key copies plus "
            f"{ENGLISH_PARTIAL_ROWS} rows of a partial "
            f"{ENGLISH_FULL_COPIES + 1}th copy"
        ),
        "sinhala_source_sha256": SINHALA_SHARDS_COMBINED_SHA256,
        "english_audio_source_sha256": ENGLISH_SHA256,
        "teacher_labels_sha256": TEACHER_SHA256,
        "shards": staged_shards,
    }
    (train / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validation = sinhala / "v4-validation-206-audio.parquet"
    if sha256(validation) != VALIDATION_SHA256:
        raise RuntimeError("validation bundle hash mismatch")
    (LOCAL_INPUT / validation.name).symlink_to(validation)

    config = {
        "experiment": "e006",
        "model_name": "openai/whisper-small",
        "output_dir": "runs/e006-whisper-small-wide-lora-r16-100h-teacher-replay-v4",
        "method": "lora",
        "training_rows": TRAINING_ROWS,
        "sinhala_source_rows": SINHALA_ROWS,
        "english_replay_rows": ENGLISH_ROWS,
        "english_unique_rows": len(unique_english),
        "training_bundle_sha256": E006_FINGERPRINT,
        "validation_bundle_sha256": VALIDATION_SHA256,
        "seed": 20260903,
        "resume_from_checkpoint": None,
        "stop_after_step": None,
        "skip_validation": False,
        "max_steps": MAX_STEPS,
        "learning_rate": 5e-5,
        "train_batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "save_steps": 205,
        "logging_steps": 10,
        "git_commit": None,
        "tracked_config": "configs/training/experiments/e006-wide-lora-r16-100h-teacher-replay-v4.json",
    }
    config_path = LOCAL_INPUT / "e006-job-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def main() -> None:
    runtime = one_file("whisper-small--model.safetensors").parent
    sinhala = one_file("sinhala-manifest.json").parent
    english_source = one_file("train-manifest.json").parent
    teacher_path = one_file("e004-english-teacher-labels.parquet")
    install_runtime(runtime)
    stage_model(runtime)
    config = stage_inputs(sinhala, english_source, teacher_path)
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
    if checkpoints["checkpoints"][-1]["step"] != MAX_STEPS:
        raise RuntimeError(f"final durable checkpoint is not step {MAX_STEPS}")
    predictions = OUTPUT / "sinhala-validation-predictions.parquet"
    result = {
        "state": "complete",
        "steps": MAX_STEPS,
        "training_fingerprint": E006_FINGERPRINT,
        "predictions_sha256": sha256(predictions),
        "adapter_sha256": sha256(OUTPUT / "final-adapter/adapter_model.safetensors"),
        "checkpoint_index_sha256": sha256(OUTPUT / "checkpoint-index.json"),
        "training_seconds": metadata["training_seconds"],
    }
    (WORK / "e006-kaggle-training-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
