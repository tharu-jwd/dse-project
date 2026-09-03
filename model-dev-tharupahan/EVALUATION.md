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
PYTHONPATH=src python scripts/evaluate_predictions.py \
  --predictions runs/RUN_ID/predictions.parquet \
  --output-dir runs/RUN_ID/evaluation
```

The evaluator writes scored rows, machine-readable aggregate/subgroup metrics,
95% paired bootstrap intervals, edit-operation counts, and a Markdown summary.
Its automatic error labels are triage signals, not linguistic ground truth.
Suspected reference/audio faults must be confirmed in the native-review UI.

Validation predictions may be used for recipe and checkpoint selection. Test
predictions must not be generated until the candidate, normalization version,
and decoding configuration have been frozen.
