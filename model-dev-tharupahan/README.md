# Sinhala ASR — Tharupahan

Clean, reproducible Sinhala ASR development. The superseded team implementation
was audited and removed from this branch; it remains available in Git history
and the branches where it was developed. See
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
on v4 validation with repetition-safe decoding. E002 and E003 improved Sinhala
but damaged standalone English. E004 established that replaying untouched-model
English behavior targets retains the Sinhala gain while passing the frozen
English gate: canonical English WER was 4.62% versus 4.23% untouched. Sinhala
remained unusable at 95.48% canonical WER and 37.63% canonical CER. E005 scaled
the same proven adapter and replay method to a nested 50,000-row, 60.31-hour
Sinhala-source subset for one effective epoch: Sinhala canonical WER improved
materially to 88.39% (canonical CER 32.24%; paired 95% interval excludes zero
on both), and the English-retention gate continued to pass at 4.52% canonical
WER versus 4.23% untouched, statistically equivalent to E004's English result.
Sinhala remains far above the under-10% target. Per the project's decision
rule, scale is a promising axis; E006 is expected to continue scaling the
Sinhala training subset. Full-parameter fine-tuning remains out of scope.

Historical split and model claims have already been captured in the
[historical audit](docs/audits/historical-audit.md). Current dataset
reproduction and validation commands are maintained in the
[dataset guide](docs/data/dataset.md); they do not depend on the removed legacy
directory.
