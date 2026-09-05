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
PYTHONPATH=src python scripts/training/train.py \
  --config configs/training/experiments/e001-wide-lora-r16-100-step-v4.json
```

Full runs require a reviewed `validation` split. Access to an unreviewed
`validation_candidate` split requires the explicit
`--allow-unreviewed-validation` flag and is permitted only for bounded plumbing
checks. `--smoke-train-rows` and `--smoke-validation-rows` limit row counts but
do not change the selected split. The test split is never loaded by training.

The local smoke and deliberate resume checks are:

```bash
PYTHONPATH=src python scripts/training/train.py \
  --config configs/training/diagnostics/tiny-cpu-smoke.json \
  --smoke-train-rows 1 --smoke-validation-rows 1 \
  --allow-unreviewed-validation

PYTHONPATH=src python scripts/training/train.py \
  --config configs/training/diagnostics/tiny-cpu-resume-smoke.json \
  --smoke-train-rows 2 --smoke-validation-rows 1 \
  --allow-unreviewed-validation
```

These configurations use no paid compute. The primary controlled experiment
starts from `openai/whisper-small`; `whisper-tiny` is only a pipeline fixture.

Dataset v4 is the current frozen input. Original, untrimmed audio is the
baseline. Dynamic boundary cropping is available through
`crop_training_audio` and `crop_proposals`, but the local 50-step A/B produced a
strong negative signal and no compute advantage, so it must remain disabled for
the first Whisper-small pilot. See [the audio audit](../data/audio-audit.md).

Matched local A/B fixture configurations are
`tiny-mps-v3-original-ab.json` and `tiny-mps-v3-trimmed-ab.json`. They are
diagnostic experiments, not candidate models or project accuracy baselines.

The isolated text-only label experiment is documented in
[The label-refinement audit](../audits/label-refinement-ab.md) documents this.
Its four `tiny-mps-label-*` configurations compare
original versus Bedrock-proposed training labels separately for Sinhala-only
and Latin-only rows. Dataset v3 and all validation references remained unchanged
during that completed historical diagnostic.

Full-parameter fine-tuning is out of scope. Current pilots use wide-target LoRA
over `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, and `fc2`, while the base
Whisper-small weights remain frozen.

Colab execution and storage isolation for E002 onward are defined in
[the Colab CLI policy](colab-cli.md). Google Drive is not mounted; verified
artifacts are downloaded into `reports/experiments/eNNN-...` before the
disposable runtime is stopped.

## Free-Colab wide-LoRA pilot

The first adapter pilot is intentionally bounded to 100 optimizer steps over a
deterministic 2,000-row v4 sample. The sample contains 1,925 Sinhala-only and 75
Latin-only rows, covers all 471 training speakers, and totals 2.42 audio hours.
It is a pipeline/learning-signal and throughput measurement, not a final model.

Open `notebooks/e001-whisper-small-wide-lora-r16-100-step-v4-colab.ipynb` in a
T4 Colab runtime. Its upload cell requires:

- `reports/colab/e001-train-v4-2000-audio.parquet`
- `reports/colab/v4-validation-206-audio.parquet`
- `scripts/training/run_e001_colab.py`

As a historical exception, the E001 notebook mounted Drive and wrote checkpoints under
`MyDrive/sinhala-asr/e001-whisper-small-wide-lora-r16-100-step-v4`. It trains
rank-16 wide LoRA with frozen
base weights, effective batch size 16, and learning rate `5e-5`, then generates
predictions for the same 206-row v4 validation split using the frozen baseline
decoding settings. The test split is neither packaged nor accessed.

The setup deliberately uninstalls Colab's optional `torchao` package. The
preinstalled `torchao==0.10.0` is incompatible with current PEFT and prevents
adapter injection; this pilot does not use quantization and does not need it.

## E002 preparation

E002 keeps the E001 adapter architecture and optimization settings while
increasing the deterministic nested training bundle to 10,000 rows and the
budget to 500 steps. The bundle contains all 2,000 E001 rows, 9,625
Sinhala/non-Latin rows, 375 Latin-only rows, all 471 training speakers, and
12.08 audio hours. Its local transport SHA-256 is
`33591d057a50a732173eba4dee3d91cfe6be79913510606330f65a665f3fefd3`.

Training cannot begin until the untouched and E001 models have both been
evaluated on the frozen English-retention benchmark. E002 uses disposable
Colab storage and the supervised background-job procedure; it does not reuse
the historical E001 Drive-mounted implementation.

## E003 preparation

E003 is a controlled replay experiment prompted by E002's English regression.
It retains all 10,000 E002 rows and adds 1,111 deterministic,
speaker-balanced LibriSpeech train-clean-100 rows, exactly 10.00% of the
11,111-row pool. Sinhala-source rows retain a Sinhala decoder prefix; replay
rows use an English decoder prefix. The training runner defaults missing prefix
metadata to Sinhala, preserving E002 compatibility.

LibriSpeech replay is prepared on a CPU-only disposable runtime and transferred
in four hash-verified shards. E003 reuses the ten verified E002 shards rather
than rebuilding one monolithic local bundle. Before real training, a two-step
GPU smoke must verify mixed-schema loading, per-row token prefixes, finite loss,
checkpoint packaging and resume compatibility.
