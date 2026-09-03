# Sinhala ASR — Tharupahan

Clean, reproducible Sinhala ASR development built independently from the
historical `model-development/` implementation. See [PLAN.md](PLAN.md) for the
decisions, quality gates, experiment sequence, and GPU cost controls.
The evidence-graded assessment of inherited work is in
[HISTORICAL_AUDIT.md](HISTORICAL_AUDIT.md).

## Development setup

```bash
cd model-dev-tharupahan
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,review]'
pytest
```

Dataset source, fingerprinting, audit, and native-speaker review commands are in
[DATASET.md](DATASET.md). Raw audio and generated reports remain local and are
ignored by Git.

Prediction scoring and error-analysis rules are in
[EVALUATION.md](EVALUATION.md).

## Audit existing splits

The first implemented gate builds a row-level manifest and fails if it finds
invalid samples or cross-split leakage:

```bash
sinhala-asr-prepare \
  --input train=../model-development/data/stratified/train.parquet \
  --input validation=../model-development/data/stratified/validation.parquet \
  --input test=../model-development/data/stratified/test.parquet \
  --output-dir reports/dataset-audit
```

Outputs:

- `manifest.parquet`: one auditable row per input sample
- `summary.json`: machine-readable counts, fingerprints, and violations
- `summary.md`: human-readable audit report

Use `--allow-invalid` only while investigating a failed audit. It does not hide
violations from the reports.
