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
  --config configs/training/small-full-pilot.json
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

Dataset v3 is the current frozen input. Original, untrimmed audio is the
baseline. Dynamic boundary cropping is available through
`crop_training_audio` and `crop_proposals`, but the local 50-step A/B produced a
strong negative signal and no compute advantage, so it must remain disabled for
the first Whisper-small pilot. See `AUDIO_AUDIT.md`.

Matched local A/B fixture configurations are
`tiny-mps-v3-original-ab.json` and `tiny-mps-v3-trimmed-ab.json`. They are
diagnostic experiments, not candidate models or project accuracy baselines.
