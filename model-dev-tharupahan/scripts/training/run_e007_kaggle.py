#!/usr/bin/env python3
"""Orchestrate one resumable phase of the full-data E007 Kaggle run."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
MODEL = WORK / "whisper-small"
LOCAL_INPUT = WORK / "e007-input"
SINHALA_SHA256 = "ecd5bbffee8a644fd2db83385799838ca055cc832335aaa90de00ca7db35e6f9"
ENGLISH_SHA256 = "0246f185b9f08a79e1eec91f7effb07e7912dd3199b00d78a72cfef61c54781b"
TEACHER_SHA256 = "5f45c712867f430ab2e911cfa8950634c69b5897689b082486c8c6f1fc45c0f6"
VALIDATION_SHA256 = "c7a378a115dd953c50bfe6a2c550e28b5a5cc829dee79bdb29ccde1151e46ec2"
E007_FINGERPRINT = "62c08dc692ffa1cbbf86e3d5e24185d7e673fed19604e30813c08431b5c915f7"
SINHALA_ROWS = 182_665
ENGLISH_UNIQUE_ROWS = 1_111
ENGLISH_ROWS = 20_296
TRAINING_ROWS = 202_961
MAX_STEPS = 6_342
PHASE_A_STOP = 4_077
SAVE_STEPS = 453


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


def load_teacher_english(source: Path, teacher_path: Path) -> list[dict]:
    source_manifest = json.loads((source / "train-manifest.json").read_text())
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


def stage_inputs(sinhala: Path, english_source: Path, teacher_path: Path, phase: str) -> Path:
    source_manifest = json.loads((sinhala / "sinhala-manifest.json").read_text())
    if source_manifest["shards_combined_sha256"] != SINHALA_SHA256:
        raise RuntimeError("E007 Sinhala source fingerprint mismatch")
    if source_manifest["rows"] != SINHALA_ROWS:
        raise RuntimeError("E007 Sinhala row count mismatch")

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
        staged_shards.append(dict(shard))
        total_sinhala += int(shard["rows"])
    if total_sinhala != SINHALA_ROWS:
        raise RuntimeError("staged Sinhala row count mismatch")

    english = load_teacher_english(english_source, teacher_path)
    for start in range(0, len(english), 1_000):
        rows = english[start : start + 1_000]
        name = f"english-unique-part-{start // 1000:04d}.parquet"
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
        "experiment": "e007",
        "source_sha256": E007_FINGERPRINT,
        "stored_rows": SINHALA_ROWS + ENGLISH_UNIQUE_ROWS,
        "training_rows_after_replay_expansion": TRAINING_ROWS,
        "sinhala_source_rows": SINHALA_ROWS,
        "english_unique_rows": ENGLISH_UNIQUE_ROWS,
        "english_replay_rows": ENGLISH_ROWS,
        "sinhala_source_sha256": SINHALA_SHA256,
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
        "experiment": "e007",
        "phase": phase,
        "model_name": "openai/whisper-small",
        "output_dir": "runs/e007-whisper-small-wide-lora-r16-full-v4-teacher-replay",
        "method": "lora",
        "training_rows": TRAINING_ROWS,
        "sinhala_source_rows": SINHALA_ROWS,
        "english_replay_rows": ENGLISH_ROWS,
        "english_unique_rows": ENGLISH_UNIQUE_ROWS,
        "english_replay_occurrences": ENGLISH_ROWS,
        "training_bundle_sha256": E007_FINGERPRINT,
        "validation_bundle_sha256": VALIDATION_SHA256,
        "seed": 20260903,
        "resume_from_checkpoint": None,
        "stop_after_step": PHASE_A_STOP if phase == "phase-a" else None,
        "skip_validation": phase == "phase-a",
        "max_steps": MAX_STEPS,
        "learning_rate": 5e-5,
        "train_batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "save_steps": SAVE_STEPS,
        "logging_steps": 10,
        "git_commit": os.environ["E007_SOURCE_COMMIT"],
        "tracked_config": "configs/training/experiments/e007-wide-lora-r16-full-v4-teacher-replay.json",
    }
    path = LOCAL_INPUT / f"e007-{phase}-job-config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def install_resume(config_path: Path) -> None:
    archive = one_file("checkpoint-004077.tar.gz")
    expected = os.environ.get("E007_RESUME_ARCHIVE_SHA256")
    if not expected or sha256(archive) != expected:
        raise RuntimeError("Phase-A resume archive hash mismatch")
    resume_root = WORK / "e007-resume"
    resume_root.mkdir()
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(resume_root, filter="data")
    checkpoint = resume_root / f"checkpoint-{PHASE_A_STOP}"
    if int((checkpoint / "COMPLETE").read_text().strip()) != PHASE_A_STOP:
        raise RuntimeError("Phase-A checkpoint COMPLETE marker mismatch")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if int(state["global_step"]) != PHASE_A_STOP:
        raise RuntimeError("Phase-A trainer global step mismatch")
    config = json.loads(config_path.read_text())
    config["resume_from_checkpoint"] = str(checkpoint)
    config["resume_archive_sha256"] = expected
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    phase = os.environ.get("E007_PHASE")
    if phase not in {"phase-a", "phase-b"}:
        raise RuntimeError("E007_PHASE must be phase-a or phase-b")
    runtime = one_file("whisper-small--model.safetensors").parent
    sinhala = one_file("sinhala-manifest.json").parent
    english_source = one_file("train-manifest.json").parent
    teacher_path = one_file("e004-english-teacher-labels.parquet")
    install_runtime(runtime)
    stage_model(runtime)
    config = stage_inputs(sinhala, english_source, teacher_path, phase)
    if phase == "phase-b":
        install_resume(config)
    output = WORK / f"e007-{phase}"
    environment = os.environ.copy()
    environment.update(
        {
            "SINHALA_ASR_JOB_ROOT": str(WORK / "runtime"),
            "SINHALA_ASR_INPUT_ROOT": str(LOCAL_INPUT),
            "SINHALA_ASR_OUTPUT_ROOT": str(output),
            "SINHALA_ASR_JOB_CONFIG": str(config),
            "SINHALA_ASR_MODEL": str(MODEL),
        }
    )
    runner = Path(__file__).with_name("run_e002_colab.py")
    subprocess.run([sys.executable, str(runner)], check=True, env=environment)
    status = json.loads((output / "status.json").read_text())
    checkpoints = json.loads((output / "checkpoint-index.json").read_text())
    expected_step = PHASE_A_STOP if phase == "phase-a" else MAX_STEPS
    if status != {"state": "complete", "exit_code": 0}:
        raise RuntimeError(f"unexpected final status: {status}")
    if checkpoints["checkpoints"][-1]["step"] != expected_step:
        raise RuntimeError(f"final durable checkpoint is not step {expected_step}")
    result = {
        "state": "complete",
        "phase": phase,
        "steps": expected_step,
        "training_fingerprint": E007_FINGERPRINT,
        "adapter_sha256": sha256(output / "final-adapter/adapter_model.safetensors"),
        "checkpoint_index_sha256": sha256(output / "checkpoint-index.json"),
        "resume_checkpoint_name": f"checkpoint-{expected_step:06d}.tar.gz",
        "resume_checkpoint_sha256": sha256(
            output / f"checkpoint-{expected_step:06d}.tar.gz"
        ),
    }
    predictions = output / "sinhala-validation-predictions.parquet"
    if predictions.exists():
        result["predictions_sha256"] = sha256(predictions)
    (WORK / f"e007-{phase}-kaggle-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
