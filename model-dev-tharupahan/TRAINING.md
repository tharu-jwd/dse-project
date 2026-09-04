# Training Operations

Install the training dependencies with `pip install -e '.[train]'`. All runs
use a checked JSON configuration and write their resolved config, Git commit,
manifest hash, platform, framework version, row counts, measured runtime, and
actual compute cost into the run directory.

The entry point rejects unknown configuration fields and rejects a projected
cost above `maximum_cost_usd`. Planned cost includes the configured safety
margin. Set the provider's displayed hourly price and a measured duration
estimate after the capped pilot; do not guess these values before renting.

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/training/small-wide-lora-pilot.json
```

Full runs require a reviewed `validation` split. Access to an unreviewed
`validation_candidate` split requires the explicit
`--allow-unreviewed-validation` flag and is permitted only for bounded plumbing
checks. `--smoke-train-rows` and `--smoke-validation-rows` limit row counts but
do not change the selected split. The test split is never loaded by training.

The local smoke and deliberate resume checks are:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/training/tiny-cpu-smoke.json \
  --smoke-train-rows 1 --smoke-validation-rows 1 \
  --allow-unreviewed-validation

PYTHONPATH=src python scripts/train.py \
  --config configs/training/tiny-cpu-resume-smoke.json \
  --smoke-train-rows 2 --smoke-validation-rows 1 \
  --allow-unreviewed-validation
```

These configurations use no paid compute. The primary controlled experiment
starts from `openai/whisper-small`; `whisper-tiny` is only a pipeline fixture.

Dataset v4 is the current frozen input. Original, untrimmed audio is the
baseline. Dynamic boundary cropping is available through
`crop_training_audio` and `crop_proposals`, but the local 50-step A/B produced a
strong negative signal and no compute advantage, so it must remain disabled for
the first Whisper-small pilot. See `AUDIO_AUDIT.md`.

Matched local A/B fixture configurations are
`tiny-mps-v3-original-ab.json` and `tiny-mps-v3-trimmed-ab.json`. They are
diagnostic experiments, not candidate models or project accuracy baselines.

The isolated text-only label experiment is documented in
`LABEL_REFINEMENT_AB.md`. Its four `tiny-mps-label-*` configurations compare
original versus Bedrock-proposed training labels separately for Sinhala-only
and Latin-only rows. Dataset v3 and all validation references remained unchanged
during that completed historical diagnostic.

Full-parameter fine-tuning is out of scope. Current pilots use wide-target LoRA
over `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, and `fc2`, while the base
Whisper-small weights remain frozen.

## Free-Colab wide-LoRA pilot

The first adapter pilot is intentionally bounded to 100 optimizer steps over a
deterministic 2,000-row v4 sample. The sample contains 1,925 Sinhala-only and 75
Latin-only rows, covers all 471 training speakers, and totals 2.42 audio hours.
It is a pipeline/learning-signal and throughput measurement, not a final model.

Open `notebooks/whisper-small-wide-lora-pilot.ipynb` in a T4 Colab runtime. Its
upload cell requires:

- `reports/colab/v4-training-pilot-2000.parquet`
- `reports/colab/v4-validation-colab.parquet`
- `scripts/colab_wide_lora_pilot.py`

The notebook mounts Drive and writes checkpoints under
`MyDrive/sinhala-asr/wide-lora-100-v1`. It trains rank-16 wide LoRA with frozen
base weights, effective batch size 16, and learning rate `5e-5`, then generates
predictions for the same 206-row v4 validation split using the frozen baseline
decoding settings. The test split is neither packaged nor accessed.

The setup deliberately uninstalls Colab's optional `torchao` package. The
preinstalled `torchao==0.10.0` is incompatible with current PEFT and prevents
adapter injection; this pilot does not use quantization and does not need it.
