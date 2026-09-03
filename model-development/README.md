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
├── notebooks/      # Exploratory preprocessing notebooks
├── observation/    # Early data exploration and collection utilities
├── data/           # Local datasets (large Parquet files are ignored)
├── checkpoints/    # Local model artifacts (ignored except .gitkeep)
├── docs/           # Guides, strategy, dataset notes, and design reviews
├── diagrams/       # Editable Mermaid diagrams and rendered copies
└── requirements.txt
```

## Start here

- [Fine-tuning guide](docs/fine-tuning-guide.md): setup, training, evaluation,
  and error-analysis commands.
- [Fine-tuning strategy](docs/fine-tuning-strategy.md): prioritized experiments
  for reducing Sinhala WER.
- [Dataset notes](docs/dataset.md): dataset provenance and preparation notes.
- [Backend integration](docs/integration-points.md): how an ASR checkpoint is
  connected to the application.
- [Experiment tracker](experiments/finetune_tracker.csv): completed runs and
  their validation, test, and forgetting metrics.

Run active ML commands from this directory so examples and relative output
paths remain consistent.
