#!/usr/bin/env python3
"""Exercise E015 data transport, mixed-language labels, and resume on Kaggle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e015_kaggle as e015


def main() -> None:
    output = e015.WORK / "e015-runtime-smoke"
    output.mkdir(parents=True, exist_ok=True)
    result_path = e015.WORK / "e015-runtime-smoke-result.json"
    try:
        runtime = e015.one_file("whisper-small--model.safetensors").parent
        source_root = e015.one_file("sinhala-manifest.json").parent
        e015_root = e015.one_file("v5-train-allowlist.parquet").parent
        source_package = e015.one_file("train.py").parent
        e015.install_runtime(runtime)
        e015.stage_model(runtime)
        e015.verify_e015_assets(e015_root)

        combined = e015.WORK / "e015-smoke-source-manifest.parquet"
        e015.build_manifest(source_root, e015_root, combined)
        rows = pq.read_table(combined).to_pylist()
        train = [row for row in rows if row["dataset_split"] == "train"]
        validation = [row for row in rows if row["dataset_split"] == "validation"]
        smoke_rows = train[:2] + train[-2:] + validation[:2]
        if {row["decoder_language"] for row in smoke_rows[:4]} != {"si", "en"}:
            raise RuntimeError("smoke manifest must contain Sinhala and English targets")
        smoke_manifest = output / "mixed-language-manifest.parquet"
        pq.write_table(pa.Table.from_pylist(smoke_rows), smoke_manifest, compression="zstd")

        run_dir = output / "train"
        common = {
            "model_name": str(e015.MODEL),
            "manifest": str(smoke_manifest),
            "output_dir": str(run_dir),
            "method": "full",
            "max_steps": 2,
            "learning_rate": 5e-5,
            "lr_scheduler_type": "linear",
            "warmup_steps": 0,
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "eval_steps": 2,
            "save_steps": 2,
            "logging_steps": 1,
            "seed": 20260903,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "hourly_price_usd": 0.0,
            "estimated_hours": 0.0,
            "maximum_cost_usd": 0.0,
        }
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_package)
        train_script = source_package / "train.py"
        phase_a = {**common, "stop_after_step": 1, "resume_from_checkpoint": None}
        phase_a_path = output / "phase-a-config.json"
        e015.write_json(phase_a_path, phase_a)
        subprocess.run(
            [sys.executable, str(train_script), "--config", str(phase_a_path)],
            check=True,
            env=environment,
        )
        checkpoint = run_dir / "checkpoint-1"
        first_inventory = e015.checkpoint_inventory(checkpoint)
        phase_b = {
            **common,
            "stop_after_step": None,
            "resume_from_checkpoint": str(checkpoint),
        }
        phase_b_path = output / "phase-b-config.json"
        e015.write_json(phase_b_path, phase_b)
        subprocess.run(
            [sys.executable, str(train_script), "--config", str(phase_b_path)],
            check=True,
            env=environment,
        )
        final_state = json.loads((run_dir / "checkpoint-2/trainer_state.json").read_text())
        if int(final_state["global_step"]) != 2:
            raise RuntimeError("resume smoke did not reach step 2")
        e015.write_json(
            result_path,
            {
                "state": "complete",
                "phase_a_global_step": first_inventory["global_step"],
                "phase_b_global_step": 2,
                "decoder_languages": sorted(
                    {row["decoder_language"] for row in smoke_rows[:4]}
                ),
                "source_manifest_sha256": e015.sha256(combined),
                "smoke_manifest_sha256": e015.sha256(smoke_manifest),
            },
        )
    except Exception as error:
        e015.write_json(
            result_path,
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
