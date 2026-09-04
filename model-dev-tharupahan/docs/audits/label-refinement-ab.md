# Transcript Label Refinement A/B

## Question

Does text-only spelling refinement improve a short Whisper training run when the
audio, sample IDs, initialization, ordering, hyperparameters, and evaluation
references are held fixed? Sinhala-only and Latin-only labels are tested as two
separate experiments.

This is a diagnostic experiment, not a dataset correction pass. A language
model cannot hear the source audio. A plausible written correction can therefore
be a worse ASR label when the speaker actually pronounced the original form.

## Earlier work

The earlier 2,000-row GPT-assisted pass covered validation/test review
candidates. It did not refine training labels and was not this A/B test.

## Construction

- Source: frozen dataset v3 training split only.
- Selection: the first 500 deterministic sample IDs in each of
  `sinhala_only` and `latin_only`.
- Refiner: `global.anthropic.claude-sonnet-4-6` through AWS Bedrock.
- Prompt: spelling, Unicode, and word-boundary corrections only; no grammar,
  translation, style rewriting, or guessed pronunciation.
- Bedrock usage: 55,803 input tokens and 9,407 output tokens (65,210 total).
- Candidate changes: 24/500 Sinhala and 18/500 Latin-only. High- and
  medium-confidence suggestions are included only in the experimental refined
  arms.
- Evaluation: 100 frozen Sinhala validation rows and all 36 frozen Latin-only
  validation rows. Validation references are byte-for-byte identical between
  each pair and were never sent for refinement.
- Training fixture: official `openai/whisper-tiny`, 50 steps, seed 20260903.
  This cheap local run can reject a harmful idea but cannot establish final
  Whisper-small accuracy.

The quality gate found some semantically inferred suggestions even among
high-confidence results—for example, title/name normalization and Sinhala word
replacement. These remain isolated in the B arm so the test measures their
combined effect. They must not be applied to dataset v3 without listening to
the corresponding audio.

## Reproduction

Export the fixed training subset, run the resumable Bedrock refiner, then build
the four matched manifests:

```bash
PYTHONPATH=src python scripts/refinement/export_label_refinement_batch.py \
  --manifest data/versions/v3/manifest.parquet \
  --output reports/label-refinement/ab-500-si-500-en-input.json \
  --sinhala 500 --english 500

PYTHONPATH=src python scripts/refinement/run_bedrock_refinement_batches.py \
  --input reports/label-refinement/ab-500-si-500-en-input.json \
  --output-dir reports/label-refinement/sonnet-ab-500-si-500-en \
  --model global.anthropic.claude-sonnet-4-6 --batch-size 50

PYTHONPATH=src python scripts/refinement/build_label_refinement_ab.py \
  --manifest data/versions/v3/manifest.parquet \
  --selection reports/label-refinement/ab-500-si-500-en-input.json \
  --refinements reports/label-refinement/sonnet-ab-500-si-500-en/combined.json \
  --output-dir data/experiments/label-refinement-sonnet-ab
```

Run the four configs named `tiny-mps-label-{si,en}-{original,refined}.json`.
Interpret changes in WER/CER alongside substitutions, deletions, and insertions.
No result from this tiny, short run authorizes changing the canonical corpus.

## Results

The four runs completed locally on Apple MPS at zero GPU/cloud training cost.
Combined measured training time was 7.47 minutes.

Trainer evaluation uses the configured 225-token generation cap:

| Slice | Labels | WER | CER | Eval loss |
|---|---:|---:|---:|---:|
| Sinhala | Original | 220.52% | 251.28% | 2.041 |
| Sinhala | Refined | 303.54% | 335.22% | 1.915 |
| Latin-only | Original | 79.78% | 35.46% | 2.917 |
| Latin-only | Refined | 80.90% | 34.82% | 3.826 |

Sinhala refinement increased WER by 83.02 percentage points (37.65%
relative) and CER by 83.94 points (33.41% relative), despite slightly lower
evaluation loss. This disagreement matters: lower teacher-forced loss did not
produce safer autoregressive decoding.

Latin-only strict WER increased by 1.12 points: one extra word error over 89
reference words. Strict CER decreased by 0.64 points: three fewer character
errors over 471 characters. These tiny, conflicting movements are well inside
the broad confidence intervals and are not evidence of improvement.

The separate row-level prediction path used the model's uncapped saved
generation defaults and exposed repetition instability:

| Slice | Labels | Strict WER | Strict CER | Word S/D/I | Hallucination flag |
|---|---:|---:|---:|---:|---:|
| Sinhala | Original | 336.08% | 412.64% | 292/132/1001 | 8 rows |
| Sinhala | Refined | 521.93% | 592.93% | 271/153/1789 | 10 rows |
| Latin-only | Original | 79.78% | 35.46% | 58/6/7 | 0 rows |
| Latin-only | Refined | 80.90% | 34.82% | 57/7/8 | 0 rows |

The Sinhala regression is dominated by 788 additional word insertions, not by
the spelling distinctions the refiner attempted to repair. Refined predictions
differed from original-arm predictions on 76/100 Sinhala validation rows;
Latin-only predictions differed on 21/36. With only 24 and 18 changed training
labels respectively, this shows that a severely under-trained tiny model is
highly unstable. It does not tell us whether any individual correction matches
its audio.

## Decision

Reject automatic text-only refinement as a corpus update. Do not create dataset
v4 from these suggestions. For the real Whisper-small baseline, retain dataset
v3 labels. If label cleanup is revisited, listen to each proposed change and
accept only audio-matching edits; then test a larger verified batch with a
stable decoding configuration and at least one repeated control run.
