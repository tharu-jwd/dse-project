# Sinhala ASR — Tharupahan

Clean, reproducible Sinhala ASR development built independently from the
historical `model-development/` implementation. See
[the project plan](docs/project/plan.md) for the
decisions, quality gates, experiment sequence, and GPU cost controls.
The complete documentation map is in [docs/README.md](docs/README.md).
The evidence-graded assessment of inherited work is in
[historical audit](docs/audits/historical-audit.md).

## Development setup

```bash
cd model-dev-tharupahan
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,review]'
pytest
```

Dataset source, fingerprinting, audit, and native-speaker review commands are in
[dataset guide](docs/data/dataset.md). Raw audio and generated reports remain local and are
ignored by Git.

Prediction scoring and error-analysis rules are in
[evaluation protocol](docs/evaluation/evaluation.md).
Training, resume, and cost-gate commands are in
[the training guide](docs/training/training.md).
The completed text-only label experiment and its rejected automatic-refinement
decision are in [the label-refinement audit](docs/audits/label-refinement-ab.md).
The audio-verified comparison of accessible Bedrock transcript refiners is in
[the refinement-model benchmark](docs/audits/refinement-model-benchmark.md).
The experiment index and result reports are in
[docs/experiments/README.md](docs/experiments/README.md).
The versioned Sinhala transcription rules are in
[the text policy](docs/data/text-policy.md).
The current GPU-credit allocation and experiment rationale are in
[the compute plan](docs/project/compute-plan.md).

## Current state

Dataset v4 is the current frozen experiment manifest: 182,665 unchanged v3
training rows plus 206 audio-verified validation and 186 audio-verified test
rows, with zero speaker overlap. Another 1,604 unheard evaluation-speaker rows
are explicitly `heldout_unreviewed`; they are not called gold data. Transcript
provenance is in [the review record](docs/data/review-provenance.md); audio
findings are in [the audio audit](docs/data/audio-audit.md). A controlled local
ablation found no benefit from boundary
trimming, so the first real baseline uses immutable, original audio.

Untouched `openai/whisper-small` scored 141.74% strict WER and 92.52% strict CER
on v4 validation with repetition-safe decoding. The completed 100-step
wide-LoRA pilot reduced strict WER to 114.26% while using 2.61% trainable
parameters, but outputs remain unusable. The next experiment is a larger
nested-data adapter pilot with standalone-English evaluation;
full-parameter fine-tuning is out of scope.

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
