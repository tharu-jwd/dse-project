# Sinhala ASR Development

This area contains the data-preparation, training, evaluation, and research
work used to adapt Whisper for Sinhala speech recognition.

## Directory map

```text
model-development/
├── training/       # Active full and LoRA training pipelines
├── evaluation/     # Evaluation programs and committed reference results
├── experiments/    # Experiment tracker and run metadata
├── scripts/        # Dataset preparation and quality-control utilities
├── notebooks/      # Audio and transcript preprocessing notebooks
├── research/       # Early data exploration notebooks and collection utilities
├── data/           # Local datasets (large Parquet files are ignored)
├── checkpoints/    # Local model artifacts (ignored except .gitkeep)
├── docs/           # Guides, strategy, dataset notes, and design reviews
└── requirements.txt
```

## Start here

- [Fine-tuning workflow](docs/fine-tuning.md): setup, commands, evaluation, and
  prioritized experiments for reducing Sinhala WER.
- [Dataset notes](docs/dataset.md): dataset provenance and preparation notes.
- [Backend integration](docs/integration-points.md): how an ASR checkpoint is
  connected to the application.
- [Experiment tracker](experiments/finetune_tracker.csv): completed runs and
  their validation, test, and forgetting metrics.

Run active ML commands from this directory so examples and relative output
paths remain consistent.

One dependency file at the directory root covers preparation, training, and
evaluation. `scripts/split_final_datasets.py` is the maintained split generator;
it produces both the standard stratified split and the cross-domain held-out
split.
