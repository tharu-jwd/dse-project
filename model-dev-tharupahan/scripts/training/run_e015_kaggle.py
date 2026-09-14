#!/usr/bin/env python3
"""Orchestrate one resumable phase of E015 full-parameter Whisper training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
MODEL = WORK / "whisper-small"
E007_SOURCE_SHA256 = "ecd5bbffee8a644fd2db83385799838ca055cc832335aaa90de00ca7db35e6f9"
E015_TRAIN_ROWS = 165_055
REPLAY_UNIQUE_ROWS = 1_111
REPLAY_ROWS = 18_339
TOTAL_TRAIN_ROWS = E015_TRAIN_ROWS + REPLAY_ROWS
MAX_STEPS = 5_732
PHASE_A_STOP = 3_300
WARMUP_STEPS = 573
SAVE_STEPS = 1_100
MONITOR_ROWS = 200


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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def install_runtime(runtime: Path) -> None:
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


def verify_e015_assets(root: Path) -> dict:
    index_path = root / "asset-index.json"
    index = json.loads(index_path.read_text())
    if index.get("experiment") != "e015":
        raise RuntimeError("wrong E015 asset index")
    for asset in index["assets"]:
        path = root / asset["name"]
        if path.stat().st_size != int(asset["bytes"]) or sha256(path) != asset["sha256"]:
            raise RuntimeError(f"E015 asset mismatch: {asset['name']}")
    return index


def build_manifest(e007_root: Path, e015_root: Path, output: Path) -> dict:
    source_index = json.loads((e007_root / "sinhala-manifest.json").read_text())
    if source_index["shards_combined_sha256"] != E007_SOURCE_SHA256:
        raise RuntimeError("E007 source fingerprint mismatch")
    allow_rows = pq.read_table(e015_root / "v5-train-allowlist.parquet").to_pylist()
    allowed_by_id = {str(row["sample_id"]): row for row in allow_rows}
    allowed = set(allowed_by_id)
    if len(allowed) != E015_TRAIN_ROWS:
        raise RuntimeError("v5 training allow-list size mismatch")

    records: list[dict] = []
    found: set[str] = set()
    for shard in source_index["shards"]:
        path = e007_root / shard["name"]
        if sha256(path) != shard["sha256"]:
            raise RuntimeError(f"E007 shard mismatch: {path.name}")
        rows = pq.read_table(
            path,
            columns=[
                "sample_id",
                "speaker_id",
                "language_class",
                "duration_seconds",
                "text",
                "audio_sha256",
            ],
        ).to_pylist()
        for source_row_index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            if sample_id not in allowed:
                continue
            expected = allowed_by_id[sample_id]
            if (
                str(row["speaker_id"]) != str(expected["speaker_id"])
                or str(row["text"]) != str(expected["text_canonical"])
                or str(row["audio_sha256"]) != str(expected["audio_sha256"])
            ):
                raise RuntimeError(f"E007/v5 row content mismatch: {sample_id}")
            found.add(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "speaker_id": str(row["speaker_id"]),
                    "language_class": str(row["language_class"]),
                    "duration_seconds": float(row["duration_seconds"]),
                    "text_canonical": str(row["text"]),
                    "audio_sha256": str(row["audio_sha256"]),
                    "audio_pcm_sha256": None,
                    "source_path": str(path),
                    "source_row_index": source_row_index,
                    "source_dataset": "openslr52",
                    "dataset_split": "train",
                    "decoder_language": "si",
                }
            )
    if found != allowed or len(records) != E015_TRAIN_ROWS:
        raise RuntimeError(f"E007/v5 training identity mismatch: found {len(found)}")

    replay_manifest = pq.read_table(e015_root / "teacher-replay-manifest.parquet").to_pylist()
    if len(replay_manifest) != REPLAY_UNIQUE_ROWS:
        raise RuntimeError("teacher replay row count mismatch")
    replay_audio = e015_root / "data.parquet"
    replay_manifest.sort(key=lambda row: str(row["sample_id"]))
    for occurrence in range(REPLAY_ROWS):
        row = replay_manifest[occurrence % REPLAY_UNIQUE_ROWS]
        records.append(
            {
                "sample_id": f"{row['sample_id']}:replay:{occurrence:05d}",
                "speaker_id": str(row["speaker_id"]),
                "language_class": str(row["language_class"]),
                "duration_seconds": float(row["duration_seconds"]),
                "text_canonical": str(row["text_canonical"]),
                "audio_sha256": str(row["audio_sha256"]),
                "audio_pcm_sha256": str(row["audio_pcm_sha256"]),
                "source_path": str(replay_audio),
                "source_row_index": int(row["source_row_index"]),
                "source_dataset": "librispeech_teacher_replay",
                "dataset_split": "train",
                "decoder_language": "en",
            }
        )

    validation_audio = e015_root / "v6-validation-audio.parquet"
    monitor = pq.read_table(e015_root / "v6-validation-monitor-200.parquet").to_pylist()
    if len(monitor) != MONITOR_ROWS:
        raise RuntimeError("validation monitor row count mismatch")
    for row in monitor:
        records.append(
            {
                **row,
                "source_path": str(validation_audio),
                "dataset_split": "validation",
                "decoder_language": "si",
            }
        )
    if len(records) != TOTAL_TRAIN_ROWS + MONITOR_ROWS:
        raise RuntimeError("combined manifest row count mismatch")
    pq.write_table(pa.Table.from_pylist(records), output, compression="zstd")
    return {
        "train_rows": TOTAL_TRAIN_ROWS,
        "sinhala_rows": E015_TRAIN_ROWS,
        "english_replay_rows": REPLAY_ROWS,
        "validation_monitor_rows": MONITOR_ROWS,
        "manifest_sha256": sha256(output),
    }


def checkpoint_inventory(checkpoint: Path) -> dict:
    files = []
    for path in sorted(p for p in checkpoint.rglob("*") if p.is_file()):
        files.append(
            {
                "path": str(path.relative_to(checkpoint)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    return {
        "directory": checkpoint.name,
        "global_step": int(state["global_step"]),
        "files": files,
    }


def locate_resume_checkpoint(expected_step: int) -> Path:
    candidates = [p for p in INPUTS.rglob(f"checkpoint-{expected_step}") if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one resume checkpoint, found {candidates}")
    checkpoint = candidates[0]
    index_candidates = list(checkpoint.parent.glob("checkpoint-index.json"))
    if len(index_candidates) != 1:
        raise RuntimeError("resume checkpoint index missing")
    expected = json.loads(index_candidates[0].read_text())
    actual = checkpoint_inventory(checkpoint)
    if expected != actual:
        raise RuntimeError("resume checkpoint inventory mismatch")
    return checkpoint


def main() -> None:
    phase = os.environ.get("E015_PHASE")
    if phase not in {"phase-a", "phase-b"}:
        raise RuntimeError("E015_PHASE must be phase-a or phase-b")
    output = WORK / f"e015-{phase}"
    output.mkdir(parents=True, exist_ok=True)
    result_path = WORK / f"e015-{phase}-result.json"
    try:
        runtime = one_file("whisper-small--model.safetensors").parent
        source_root = one_file("sinhala-manifest.json").parent
        e015_root = one_file("v5-train-allowlist.parquet").parent
        source_package = one_file("train.py").parent
        train_script = source_package / "train.py"
        predict_script = source_package / "predict.py"
        sinhala_asr = source_package / "sinhala_asr"
        if not predict_script.is_file() or not sinhala_asr.is_dir():
            raise RuntimeError("E015 orchestration runtime is incomplete")
        install_runtime(runtime)
        stage_model(runtime)
        asset_index = verify_e015_assets(e015_root)

        combined = WORK / "e015-combined-manifest.parquet"
        manifest_summary = build_manifest(source_root, e015_root, combined)
        run_dir = output / "train"
        resume = locate_resume_checkpoint(PHASE_A_STOP) if phase == "phase-b" else None
        boundary = PHASE_A_STOP if phase == "phase-a" else MAX_STEPS
        config = {
            "model_name": str(MODEL),
            "manifest": str(combined),
            "output_dir": str(run_dir),
            "method": "full",
            "max_steps": MAX_STEPS,
            "stop_after_step": boundary,
            "learning_rate": 5e-5,
            "lr_scheduler_type": "linear",
            "warmup_steps": WARMUP_STEPS,
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "eval_steps": SAVE_STEPS,
            "save_steps": SAVE_STEPS,
            "logging_steps": 20,
            "seed": 20260903,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "resume_from_checkpoint": str(resume) if resume else None,
            "hourly_price_usd": 0.0,
            "estimated_hours": 0.0,
            "maximum_cost_usd": 0.0,
        }
        config_path = output / "config.json"
        write_json(config_path, config)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_package)
        completed = subprocess.run(
            [sys.executable, str(train_script), "--config", str(config_path)],
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"training exited {completed.returncode}")

        boundary_checkpoint = run_dir / f"checkpoint-{boundary}"
        if not boundary_checkpoint.is_dir():
            raise RuntimeError(f"boundary checkpoint missing: {boundary_checkpoint}")
        final_model = run_dir / "final"
        predictions = output / f"e015-{phase}-monitor-predictions.parquet"
        subprocess.run(
            [
                sys.executable,
                str(predict_script),
                "--model",
                str(final_model),
                "--manifest",
                str(combined),
                "--split",
                "validation",
                "--output",
                str(predictions),
                "--batch-size",
                "8",
                "--num-beams",
                "1",
            ],
            check=True,
            env=environment,
        )

        inventory = checkpoint_inventory(boundary_checkpoint)
        write_json(run_dir / "checkpoint-index.json", inventory)
        for checkpoint in run_dir.glob("checkpoint-*"):
            if checkpoint != boundary_checkpoint:
                shutil.rmtree(checkpoint)
        shutil.rmtree(final_model)
        result = {
            "state": "complete",
            "phase": phase,
            "boundary_step": boundary,
            "manifest": manifest_summary,
            "input_asset_index_sha256": sha256(e015_root / "asset-index.json"),
            "checkpoint_index_sha256": sha256(run_dir / "checkpoint-index.json"),
            "predictions_sha256": sha256(predictions),
            "input_summary": asset_index,
        }
        write_json(result_path, result)
        print(json.dumps(result, indent=2))
    except Exception as error:
        failure = {
            "state": "failed",
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(result_path, failure)
        raise


if __name__ == "__main__":
    main()
