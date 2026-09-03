# Whisper Sinhala Fine-Tuning Guide

This directory contains the reproducible training and evaluation workflow for
adapting multilingual Whisper to Sinhala ASR. Run commands from
`model-development/`.

## Expected data layout

```text
model-development/
├── data/stratified/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
├── training/
├── evaluation/
└── requirements.txt
```

The Parquet files are ignored by Git. Copy them from GCS before an experiment:

```bash
mkdir -p data/stratified
gsutil -m cp \
  gs://singen/whisper/finalData/stratified/train.parquet \
  gs://singen/whisper/finalData/stratified/validation.parquet \
  gs://singen/whisper/finalData/stratified/test.parquet \
  data/stratified/
```

## Install and validate

```bash
cd model-development
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 training/finetune_whisper.py --smoke-test
python3 training/finetune_whisper_lora.py --smoke-test
python3 training/prepare_whisper_dataset.py data/stratified/test.parquet
```

## Full fine-tuning

```bash
python3 training/finetune_whisper.py \
  --output-dir /workspace/whisper-small-sinhala/full-lr2e-5 \
  --run-name full-lr2e-5 \
  --wandb-project whisper \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 32 \
  --num-train-epochs 6
```

## LoRA fine-tuning

```bash
python3 training/finetune_whisper_lora.py \
  --output-dir /workspace/whisper-small-sinhala-lora/wide-lora \
  --run-name wide-lora \
  --wandb-project whisper \
  --learning-rate 1e-4 \
  --per-device-train-batch-size 32 \
  --num-train-epochs 4 \
  --lora-target-modules q_proj k_proj v_proj out_proj fc1 fc2
```

Use `--help` on either script for SpecAugment, noise, time-stretch, pitch,
LoRA, checkpoint, and batch controls.

## Evaluate checkpoints

```bash
python3 evaluation/evaluate_finetuned.py \
  --model /workspace/whisper-small-sinhala/full-lr2e-5 \
  --lora /workspace/whisper-small-sinhala-lora/wide-lora:openai/whisper-small \
  --output-dir evaluation/results/generated
```

Other evaluation entry points:

```bash
python3 evaluation/evaluate_baselines.py --help
python3 evaluation/evaluate_english_forgetting.py --help
```

## Analyze errors

```bash
python3 evaluation/error_analysis.py \
  --predictions evaluation/results/generated/full-lr2e-5_predictions.csv \
  --output-dir evaluation/results/generated/error-analysis
```

Committed reference results live under `evaluation/results/`. New generated
results belong under `evaluation/results/generated/`, which is ignored by Git.

Record every run in `experiments/finetune_tracker.csv`. The prioritized
experiment ladder is in [fine-tuning-strategy.md](fine-tuning-strategy.md).
