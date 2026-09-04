# Evaluation Protocol

Every model is scored from a row-level prediction file containing at least
`sample_id`, `reference`, and `prediction`. Preserve manifest metadata such as
`source_dataset`, `language_class`, `speaker_id`, `duration_seconds`, and
`dataset_split` in the same file so subgroup reports remain traceable.

Strict scoring performs only Unicode NFC normalization and whitespace cleanup.
Canonical scoring applies the versioned metric normalization in
`src/sinhala_asr/text/normalizer.py`. CER excludes whitespace characters; WER
uses whitespace-delimited words. Report both metrics as ratios or percentages,
never as an unlabeled bare number.

```bash
PYTHONPATH=src python scripts/evaluation/predict.py \
  --model openai/whisper-small \
  --manifest data/versions/v4/manifest.parquet \
  --split validation \
  --output runs/untouched-small/validation-predictions.parquet

PYTHONPATH=src python scripts/evaluation/evaluate_predictions.py \
  --predictions runs/untouched-small/validation-predictions.parquet \
  --output-dir runs/RUN_ID/evaluation
```

Prediction refuses candidate splits unless a bounded `--max-rows` smoke limit
is supplied. It also refuses the locked test split unless `--unlock-test` is
explicitly supplied after model and decoding selection are frozen.

The evaluator writes scored rows, machine-readable aggregate/subgroup metrics,
95% paired bootstrap intervals, edit-operation counts, and a Markdown summary.
Its automatic error labels are triage signals, not linguistic ground truth.
Suspected reference/audio faults must be confirmed in the native-review UI.

Validation predictions may be used for recipe and checkpoint selection. Test
predictions must not be generated until the candidate, normalization version,
and decoding configuration have been frozen.

## English retention benchmark

Standalone English retention uses the complete 2,620-row LibriSpeech
[`test.clean` split](https://huggingface.co/datasets/openslr/librispeech_asr)
from `openslr/librispeech_asr` (CC BY 4.0), not English rows from the Sinhala
corpus. The immutable upstream Parquet SHA-256 is
`7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0`.
The prepared benchmark contains 5.403 hours from 40 speakers.

The preparation script verifies that source hash, expected row count, 16-kHz
mono audio, and non-empty embedded bytes. It records a semantic content
fingerprint over sample/speaker/chapter IDs, reference text, and each audio
hash. That content fingerprint—not a version-dependent Parquet serialization
hash—defines equality across local and Colab copies.

Every adapter is compared with untouched `openai/whisper-small` on these exact
rows using English transcription prompts and paired bootstrap intervals. This
benchmark is evaluation-only and is never mixed into Sinhala training unless a
later, separately documented English-replay experiment explicitly introduces a
licensed training split.

## Explaining metric changes

Every experiment comparison must record the run IDs, starting checkpoint,
dataset fingerprint, split, seed, preprocessing, decoding settings, and the one
intended factor changed. Report strict and canonical WER/CER for both runs,
their absolute and relative changes, paired confidence intervals, and separate
insertion, deletion, and substitution counts.

Break changes down by at least language class, duration, and transcript length;
add source, speaker, code-switch, named-entity, or other slices when relevant.
Retain sample-level improvements and regressions so the aggregate movement can
be traced to concrete utterances and error categories. If WER and CER disagree,
or one subgroup improves while another regresses, state that explicitly.

Call a result improved or regressed only when the paired evidence supports that
conclusion. A small movement inside the confidence interval is `inconclusive`,
not an improvement. Distinguish demonstrated causes from plausible mechanisms:
a controlled one-factor ablation can support attribution; an uncontrolled
comparison can only establish association. No WER/CER change may be reported
without an accompanying comparison record and explanation status.
