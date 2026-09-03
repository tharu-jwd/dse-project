# Sinhala ASR Rebuild Plan

This file is the durable source of truth for the clean Sinhala ASR pipeline.
Update it whenever a decision changes or a phase is completed. Do not rely on
chat history or undocumented notebook state.

## Objective

Build a reproducible preprocessing, fine-tuning, evaluation, and error-analysis
pipeline for Sinhala ASR. Start each controlled baseline from an official
pretrained multilingual Whisper checkpoint, initially
`openai/whisper-small`, rather than from a team fine-tuned checkpoint or random
weights.

Existing checkpoints and results are historical comparison baselines only.
`model-development/` remains read-only reference material; code is copied from
it only after review.

## Non-negotiable rules

1. Do not rent a GPU until data validation, metrics, tests, smoke tests, and
   checkpoint resume work locally.
2. Do not use the test set for model or decoding selection.
3. Do not compare runs trained on different dataset or normalization versions.
4. Every run must record its Git commit, full resolved configuration, dataset
   fingerprint, split fingerprint, seed, package versions, hardware, duration,
   planned cost, and actual cost.
5. Training and evaluation paths must be configurable; no machine-specific or
   hard-coded cloud paths.
6. Training must be resumable and important checkpoints must be synchronized to
   durable storage before a cloud instance is terminated.
7. Generated datasets, checkpoints, predictions, and reports are not committed
   unless they are deliberately selected compact reference artifacts.

## Evidence from the historical runs

The historical best full fine-tune has 17.36% WER under the repository's
current normalizer. Its error profile shows that aggregate WER mixes several
different problems:

- Raw exact-match rate: 53.77%
- Current normalized exact-match rate: 60.42%
- Exact-match rate after additionally ignoring whitespace: 76.54%
- Sinhala-only WER: 16.58% over 15,258 samples
- Code-switched WER: 29.77% over 225 samples
- WER for 1-2 word samples: 13.90%
- WER for 3-5 word samples: 14.88%
- WER for 6-10 word samples: 19.14%
- WER for 11-20 word samples: 20.30%
- WER for 21+ word samples: 35.04%

Frequent counted errors include compound-word spacing, particles joined to or
split from words, colloquial/formal variants, and spelling variants. Genuine
failures also include Sinhala grapheme confusions, names, numbers, code-switched
English, deletions, and occasional repetition loops.

Historical prediction exports lack sample ID, source, speaker, audio hash,
duration, and other metadata, so they cannot support source- or speaker-level
diagnosis. The new pipeline must retain these fields.

## Target structure

```text
model-dev-tharupahan/
├── PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── data/
│   ├── training/
│   └── evaluation/
├── src/sinhala_asr/
│   ├── data/
│   │   ├── ingest.py
│   │   ├── validate.py
│   │   ├── normalize.py
│   │   ├── deduplicate.py
│   │   ├── split.py
│   │   └── manifest.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── collator.py
│   │   ├── augment.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── inference.py
│   │   ├── metrics.py
│   │   ├── error_analysis.py
│   │   └── reports.py
│   └── text/
│       └── sinhala_normalizer.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── analyze_errors.py
├── tests/
├── data/          # ignored
├── checkpoints/   # ignored
├── reports/       # generated outputs ignored by default
└── runs/          # resolved run records ignored by default
```

The exact module split may be simplified when implementation shows that two
modules do not have genuinely separate responsibilities.

## Phase 1: dataset audit and manifest

Build a deterministic manifest containing at least:

- Stable sample ID
- Source dataset
- Source record ID
- Speaker ID when available
- Audio content hash
- Transcript hash
- Duration, sample rate, channels, and encoding
- Original transcript
- Canonical transcript
- Code-switch flag
- Split assignment
- Validation flags and exclusion reason

Produce reports for corrupt or missing audio, silence, clipping, duration
outliers, Unicode anomalies, exact and near duplicates, repeated transcripts,
speaker leakage, audio leakage, and source/speaker/split distributions.

Exit with failure when invariants are violated. Never silently discard a row;
write its reason to an exclusions manifest.

## Phase 2: Sinhala text policy

Preserve three representations:

- `text_original`: immutable source transcript
- `text_canonical`: consistent training target
- `text_metric`: explicitly normalized scoring representation

The policy must document Unicode NFC, whitespace, punctuation, numbers, dates,
abbreviations, Sinhala compounds and particles, colloquial/formal variants,
English words, and transliteration. Normalization must be deterministic,
versioned, and unit-tested with Sinhala examples.

Never hide recognition failures by over-normalizing. Report strict and
canonical metrics together.

## Phase 3: evaluation and error analysis

Every prediction row must retain manifest metadata and include reference,
prediction, strict/canonical scores, and error labels.

Required aggregate views:

- Strict WER and CER
- Canonical WER and CER
- Sinhala-only and code-switched WER
- WER by source, speaker, duration, and transcript length
- Named-entity, number, and domain-term performance when labeled
- Substitution, deletion, and insertion rates
- Empty output, truncation, repetition, and hallucination indicators
- Bootstrap confidence intervals for headline metrics

Required sample-level error taxonomy:

- Punctuation only
- Whitespace or compound segmentation
- Accepted spelling variant
- Colloquial/formal mismatch
- Sinhala grapheme confusion
- Acoustic substitution
- Deletion
- Insertion
- Named entity
- Number
- Code-switch
- Truncation
- Repetition or hallucination
- Suspected bad reference
- Suspected bad audio

Generate machine-readable tables and a human-readable Markdown or HTML report.

## Phase 4: training

Begin with one reproducible `openai/whisper-small` baseline. Support full and
LoRA training behind the same configuration schema. Track strict/canonical
validation metrics, loss, learning rate, gradient norm, throughput, peak memory,
checkpoint identity, and wall time.

Initial controlled experiments after baseline reproduction:

1. Full fine-tune with cosine scheduling at `1e-5`, `2e-5`, and `3e-5`.
2. Augmentation ablation: none, SpecAugment, speed perturbation, and
   deployment-matched real noise.
3. Wide-target LoRA comparison.
4. Whisper-medium using only the winning small-model recipe.
5. Domain adaptation and filtered pseudo-labeling only after earlier phases.

Change one experimental factor at a time. Use validation data and early stopping
for selection; evaluate the test set only after freezing a candidate.

## Phase 5: GPU and cloud cost gates

### Gate A: free/local checks

- Unit and integration tests pass
- Dataset audit passes
- Metrics pass fixed fixtures
- CPU smoke test completes
- A checkpoint can be saved, loaded, and resumed
- A small prediction report is generated successfully

### Gate B: capped GPU smoke test

- Use a fixed step limit and small data subset
- Verify mixed precision and peak VRAM
- Verify data-loader throughput and GPU utilization
- Verify checkpoint upload and recovery
- Deliberately interrupt and resume once
- Verify automatic cleanup and shutdown behavior

### Gate C: measured pilot

Run 5-10% of the training data or 500-1,000 steps. Record samples/second,
evaluation time, checkpoint time, peak VRAM, and total billed time. Estimate:

```text
planned cost = hourly GPU price × estimated hours × 1.25 safety margin
```

Do not start a full run without an approved maximum cost and stop condition.

### Gate D: bounded full experiments

- Run one baseline first
- Advance only the best one or two short pilots
- Run one full winner
- Attempt a larger model only when the measured expected benefit justifies cost

Use cloud object storage for durable data and checkpoints. Do not use a stopped
GPU volume as long-term storage. Record the instance type and displayed hourly
price at run start, and terminate compute automatically after success or error.

## Completion criteria

The project is ready for paid full training only when:

- A frozen, fingerprinted, leakage-checked dataset exists
- Sinhala normalization is documented and tested
- Untouched Whisper baseline results are recorded
- Detailed error reports work end to end
- Training smoke and resume tests pass
- Pilot throughput and cost are measured
- The full run config and maximum spend are approved

The project is complete when a selected checkpoint has reproducible strict and
canonical test results, subgroup/error analysis, English-retention results when
required, deployment latency measurements, a model card, and a documented
actual cloud cost.

## Progress

- [x] Historical run and error-profile review
- [x] Clean-project scope and architecture recorded
- [x] Package scaffold and development tooling
- [x] Dataset manifest and initial audit
- [x] Conservative Sinhala normalization v1 and tests
- [ ] Leakage-resistant split generation
- [ ] Evaluation and detailed error reports
- [ ] Configuration-driven training
- [ ] Local smoke and checkpoint-resume tests
- [ ] Capped GPU pilot and cost estimate
- [ ] Approved full baseline
- [ ] Controlled experiments
- [ ] Final test, deployment benchmark, and model card
