# Evaluation Results

This directory keeps compact, reviewable evidence from completed ASR runs:

- `error_analysis/comparison_summary.csv` contains aggregate error rates.
- `*_confusions.txt` records common substitutions, deletions, and insertions.
- `*_clusters.txt` provides compact thematic samples of model failures.

Full prediction exports and per-sample severity reports are reproducible from
the evaluation scripts and can contain tens of thousands of rows. They are
ignored by Git; generate them locally under `generated/` when performing a new
analysis.
