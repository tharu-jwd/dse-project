"""Shared MLflow helpers for finetune_whisper.py and finetune_whisper_lora.py.

Centralizes three things so both scripts stay identical in how they use
MLflow:
  1. Pointing at the shared tracking server (MLFLOW_TRACKING_URI env var).
  2. Warning -- not blocking -- when the exact hyperparameter combination
     you're about to run has already been logged by someone else, so you
     don't burn GPU hours repeating a teammate's sweep.
  3. A consistent params dict (build_run_params) so "same hyperparameters"
     means the same thing across both the full fine-tune and LoRA scripts.

MLFLOW_TRACKING_URI is not hardcoded here -- set it once per machine (shell
profile, or export it before running the script) to point at the shared
server, e.g.:
    export MLFLOW_TRACKING_URI=http://<tracking-server-host>:5000
If it's unset, MLflow falls back to a local ./mlruns folder next to wherever
the script is run from -- runs still get tracked, just not shared with
anyone else. See final-scripts/README.md for the full setup.
"""

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT = "whisper-sinhala-finetune"

# Compared, as exact-match strings, to decide whether two runs used "the
# same hyperparameters". Deliberately excludes free-text/identifying fields
# (run_name, output_dir, wandb_project) that vary per run on purpose.
DUPLICATE_CHECK_KEYS = [
    "training_type", "model_name",
    "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
    "num_train_epochs", "warmup_steps", "spec_augment",
    "noise_prob", "noise_snr_db_min", "noise_snr_db_max",
    "stretch_prob", "stretch_rate_min", "stretch_rate_max",
    "pitch_prob", "pitch_semitones_min", "pitch_semitones_max",
    "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules",
]


def _format_param(value):
    """Canonicalize a value the same way for comparison and for
    mlflow.log_params -- floats always render the same regardless of how
    they were spelled on the command line (3e-5, 0.00003, ... -> '3e-05').
    """
    if isinstance(value, float):
        return str(float(value))
    return str(value)


def build_run_params(args, training_type):
    """Flatten the argparse Namespace into the params dict logged to MLflow
    and used for the duplicate-run check. `training_type` is "full" or
    "lora" (LoRA-only args are simply absent from `args` for the full
    fine-tune script, so this works unchanged for both).
    """
    skip = {"output_dir", "run_name", "wandb_project", "smoke_test",
            "mlflow_experiment", "use_mlflow", "mlflow_log_artifacts"}
    params = {k: v for k, v in vars(args).items() if k not in skip}
    params["training_type"] = training_type
    params["spec_augment"] = not args.no_spec_augment
    return {k: _format_param(v) for k, v in params.items()}


def setup(experiment_name=None):
    """Select (creating if needed) the MLflow experiment. Tracking URI comes
    from MLFLOW_TRACKING_URI in the environment -- see module docstring.
    Returns the resolved tracking URI, for a one-line log message.
    """
    mlflow.set_experiment(experiment_name or DEFAULT_EXPERIMENT)
    return mlflow.get_tracking_uri()


def warn_if_duplicate(experiment_name, params):
    """Search the experiment for prior runs matching `params` on every key
    in DUPLICATE_CHECK_KEYS, and print a warning listing them if found.
    Never raises and never blocks -- it's a heads-up, not a gate, since a
    deliberate re-run (different seed, verifying a result) is legitimate.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name or DEFAULT_EXPERIMENT)
    if experiment is None:
        return []  # nothing logged in this experiment yet

    compare = {k: params[k] for k in DUPLICATE_CHECK_KEYS if k in params}
    filter_string = " and ".join(f"params.{k} = '{v}'" for k, v in compare.items())
    matches = client.search_runs(
        [experiment.experiment_id], filter_string=filter_string, max_results=5,
    )
    if matches:
        print("\n" + "=" * 70)
        print(f"MLflow: {len(matches)} existing run(s) already used this exact "
              f"hyperparameter combination:")
        for run in matches:
            name = run.data.tags.get("mlflow.runName", run.info.run_id)
            wer = run.data.metrics.get("eval_wer")
            wer_str = f"eval_wer={wer:.2f}" if wer is not None else "eval_wer=(run did not finish)"
            print(f"  - {name}  ({wer_str})  run_id={run.info.run_id}")
        print("Check the MLflow UI before continuing -- you may be duplicating "
              "a teammate's run. Proceeding anyway (this is a warning, not a block).")
        print("=" * 70 + "\n")
    return matches
