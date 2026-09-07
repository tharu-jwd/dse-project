#!/usr/bin/env python3
"""E012: cheap grid search for full-fine-tune learning rate x replay ratio.

Runs scripts/training/train.py (method="full") once per trial, in a fresh
subprocess each time -- same reasoning as scripts/training/optuna_search.py's
own docstring: GPU memory from one trial's Trainer instance must not leak
into the next. 9 trials total: 3 learning rates (log-spaced across the
~1e-6 to 5e-5 range documented in docs/project/plan.md item 7 -- LoRA's
winning 2.345e-4 does not transfer to full-parameter updates) x 3 replay
ratios (10%/20%/30% -- the project's 10% figure was a one-time LoRA-era
design choice, never itself validated; see plan.md item 7 and
docs/data/dataset.md). 100 steps/trial, batch 4 x accumulation 4 = 1,600
occurrences, matching E008's own per-trial pilot scale.

Unlike E008 (which only ever logged eval_wer), this also reads eval_loss
from each trial's Sinhala-only validation slice and separately reports the
per-trial train_loss trend on the replay rows, as an early, cheap read on
whether a config is trending toward forgetting -- not a substitute for the
real English-retention check (LibriSpeech-benchmark scoring), which still
belongs on the narrowed winner only (see plan.md item 7 step 3), not on all
9 pilot-scale trials.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
MODEL = WORK / "whisper-small"
RUNTIME_DIR = WORK / "runtime"

LEARNING_RATES = [1e-6, 5e-6, 5e-5]
REPLAY_RATIOS = [10, 20, 30]


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
    import shutil

    shutil.rmtree(MODEL, ignore_errors=True)
    MODEL.mkdir(parents=True)
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    if not (MODEL / "model.safetensors").is_file():
        raise RuntimeError("offline Whisper-small model is incomplete")


def read_trainer_state(output_dir: Path) -> dict:
    state_path = output_dir / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"no trainer_state.json in {output_dir}; trial did not complete")
    log_history = json.loads(state_path.read_text(encoding="utf-8"))["log_history"]
    eval_entries = [entry for entry in log_history if "eval_wer" in entry]
    train_entries = [entry for entry in log_history if "loss" in entry and "eval_wer" not in entry]
    if not eval_entries:
        raise RuntimeError(f"no eval_wer logged in {state_path}")
    return {
        "eval_wer": eval_entries[-1]["eval_wer"],
        "eval_cer": eval_entries[-1]["eval_cer"],
        "eval_loss": eval_entries[-1]["eval_loss"],
        "final_train_loss": train_entries[-1]["loss"] if train_entries else None,
    }


def main() -> None:
    runtime = one_file("whisper-small--model.safetensors").parent
    train_script = one_file("train.py")
    sinhala_asr_src = one_dir("sinhala_asr")

    install_runtime(runtime)
    stage_model(runtime)

    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = str(sinhala_asr_src.parent)

    results = []
    # The three search manifests store source_path as relative paths
    # ("data/raw/<name>/data.parquet") matching their local repo layout at
    # build time (see scripts/data/build_full_finetune_search_manifests.py).
    # The Kaggle inputs dataset preserves that same relative structure, so
    # running train.py with cwd set to this dataset's mount root resolves
    # them correctly without rewriting any manifest content.
    search_inputs_root = one_file("manifest-replay10.parquet").parent

    for replay_ratio in REPLAY_RATIOS:
        manifest = one_file(f"manifest-replay{replay_ratio:02d}.parquet")
        for learning_rate in LEARNING_RATES:
            trial_id = f"replay{replay_ratio:02d}-lr{learning_rate:.0e}"
            trial_dir = WORK / "trials" / trial_id
            config = {
                "model_name": str(MODEL),
                "manifest": str(manifest),
                "output_dir": str(trial_dir),
                "method": "full",
                "max_steps": 100,
                "learning_rate": learning_rate,
                "train_batch_size": 4,
                "eval_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "eval_steps": 100,
                "save_steps": 100,
                "logging_steps": 10,
                "fp16": True,
                "bf16": False,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 2,
                "hourly_price_usd": 0.0,
                "estimated_hours": 0.0,
                "maximum_cost_usd": 0.0,
            }
            trial_dir.mkdir(parents=True, exist_ok=True)
            config_path = trial_dir / "config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

            print(f"=== trial {trial_id} ===", flush=True)
            log_path = trial_dir / "train.log"
            with log_path.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    [sys.executable, str(train_script), "--config", str(config_path)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=subprocess_env,
                    cwd=str(search_inputs_root),
                )
            if completed.returncode != 0:
                print(f"trial {trial_id} FAILED, see {log_path}", flush=True)
                results.append({"trial": trial_id, "replay_ratio": replay_ratio,
                                 "learning_rate": learning_rate, "status": "failed"})
            else:
                metrics = read_trainer_state(trial_dir)
                row = {"trial": trial_id, "replay_ratio": replay_ratio,
                       "learning_rate": learning_rate, "status": "complete", **metrics}
                results.append(row)
                print(json.dumps(row, indent=2), flush=True)

            # Full-parameter checkpoints are ~3GB/trial (model.safetensors +
            # AdamW optimizer states), unlike LoRA's few-MB adapters -- the
            # first attempt at this search filled Kaggle's working disk by
            # trial 5 and crashed mid-write (torch.serialization RuntimeError:
            # "basic_ios::clear: iostream error"). Only trainer_state.json/
            # all_results.json/train.log are needed for the search decision,
            # so delete the actual model weights immediately after reading them.
            for checkpoint_dir in trial_dir.glob("checkpoint-*"):
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            shutil.rmtree(trial_dir / "final", ignore_errors=True)

    output_path = WORK / "e012-search-results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("=== ALL TRIALS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
