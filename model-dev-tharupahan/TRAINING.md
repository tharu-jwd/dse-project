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

Full runs require a reviewed `validation` split. The unreviewed
`validation_candidate` split is accepted only when a bounded
`--smoke-validation-rows` argument is present. The test split is never loaded by
the training entry point.

The local smoke and deliberate resume checks are:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/training/tiny-cpu-smoke.json \
  --smoke-train-rows 1 --smoke-validation-rows 1

PYTHONPATH=src python scripts/train.py \
  --config configs/training/tiny-cpu-resume-smoke.json \
  --smoke-train-rows 2 --smoke-validation-rows 1
```

These configurations use no paid compute. The primary controlled experiment
starts from `openai/whisper-small`; `whisper-tiny` is only a pipeline fixture.
