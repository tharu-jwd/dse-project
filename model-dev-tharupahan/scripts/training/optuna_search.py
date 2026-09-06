#!/usr/bin/env python3
"""Automated LoRA rank/learning-rate search via Optuna.

Runs the existing local training entry point (train.py) once per trial, in a
fresh subprocess each time -- not in-process -- so GPU/MPS memory from one
trial's Trainer instance cannot leak or fragment into the next. Each trial
gets its own resolved config (a copy of --base-config with lora_rank,
lora_alpha, and learning_rate overridden by Optuna's suggestion) and its own
output directory. The objective is the trial's final validation WER, read
back from the standard trainer_state.json log_history train.py already
writes -- no changes to what a run records.

lora_alpha is fixed to 2x the suggested lora_rank (the standard LoRA
convention) rather than searched independently, keeping the search space to
the two axes this project's plan actually calls out: adapter target-width
(rank) and learning rate.

The Optuna study is stored in a local SQLite file, so a search can be
interrupted and resumed with the same --study-name/--storage without losing
completed trials.

Local, free smoke test (matches Gate A -- prove the harness works before ever
pointing it at a real pilot):

    PYTHONPATH=src python scripts/training/optuna_search.py \\
      --base-config configs/training/diagnostics/tiny-cpu-lora-smoke.json \\
      --study-name smoke-test --storage sqlite:///runs/optuna-smoke/study.db \\
      --output-dir runs/optuna-smoke --n-trials 2 \\
      --smoke-train-rows 2 --smoke-validation-rows 1 --allow-unreviewed-validation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "training" / "train.py"


def read_final_eval_wer(output_dir: Path) -> float:
    state_path = output_dir / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"no trainer_state.json in {output_dir}; trial did not complete")
    log_history = json.loads(state_path.read_text(encoding="utf-8"))["log_history"]
    eval_entries = [entry for entry in log_history if "eval_wer" in entry]
    if not eval_entries:
        raise RuntimeError(f"no eval_wer logged in {state_path}")
    return float(eval_entries[-1]["eval_wer"])


def run_trial(
    trial: optuna.Trial,
    base_config: dict,
    output_root: Path,
    passthrough_args: list[str],
) -> float:
    lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    lora_alpha = 2 * lora_rank

    trial_dir = output_root / f"trial-{trial.number:03d}"
    trial_config = {
        **base_config,
        "output_dir": str(trial_dir),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
    }
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.json"
    config_path.write_text(json.dumps(trial_config, indent=2) + "\n", encoding="utf-8")

    log_path = trial_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path), *passthrough_args],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"trial {trial.number} training subprocess failed (exit {result.returncode}); "
            f"see {log_path}"
        )
    return read_final_eval_wer(trial_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--storage", required=True, help="e.g. sqlite:///runs/optuna/study.db")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, required=True)
    parser.add_argument("--smoke-train-rows", type=int)
    parser.add_argument("--smoke-validation-rows", type=int)
    parser.add_argument("--allow-unreviewed-validation", action="store_true")
    args = parser.parse_args()

    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    passthrough_args: list[str] = []
    if args.smoke_train_rows:
        passthrough_args += ["--smoke-train-rows", str(args.smoke_train_rows)]
    if args.smoke_validation_rows:
        passthrough_args += ["--smoke-validation-rows", str(args.smoke_validation_rows)]
    if args.allow_unreviewed_validation:
        passthrough_args += ["--allow-unreviewed-validation"]

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(
        lambda trial: run_trial(trial, base_config, output_root, passthrough_args),
        n_trials=args.n_trials,
    )

    print(f"Completed {len(study.trials)} total trial(s).")
    print(f"Best trial: #{study.best_trial.number}, WER={study.best_value:.4f}, params={study.best_params}")

    summary = {
        "best_trial_number": study.best_trial.number,
        "best_wer": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    (output_root / "search-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
