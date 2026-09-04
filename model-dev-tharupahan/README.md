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
Training, resume, and cost-gate commands are in [TRAINING.md](TRAINING.md).
The completed text-only label experiment and its rejected automatic-refinement
decision are in [LABEL_REFINEMENT_AB.md](LABEL_REFINEMENT_AB.md).
The versioned Sinhala transcription rules are in [TEXT_POLICY.md](TEXT_POLICY.md).
The current GPU-credit allocation and experiment rationale are in
[COMPUTE_PLAN.md](COMPUTE_PLAN.md).

## Current state

Dataset v3 is the frozen training source: 182,665 train, 1,000 validation, and
999 locked test rows, with zero speaker overlap. Transcript provenance is in
`REVIEW_PROVENANCE.md`; audio findings are in `AUDIO_AUDIT.md`. A controlled
local ablation found no benefit from boundary trimming, so the first real
baseline uses immutable, original audio.

The v4 native-listening review is now rebuilding trustworthy evaluation
references from normalized OpenSLR text. It covers all 295 disputed v3
references plus 100 unchanged controls; v3 remains immutable until that review
is complete.

The next experiment is untouched `openai/whisper-small` inference on v3
validation, followed by the detailed comparison protocol in `EVALUATION.md`.

## Audit historical splits

For historical investigation, this builds a row-level manifest and fails if it finds
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
