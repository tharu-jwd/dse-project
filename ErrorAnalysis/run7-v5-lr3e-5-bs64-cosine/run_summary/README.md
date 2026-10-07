# run7-v5-lr3e-5-bs64-cosine

- **W&B run:** [run7-v5-lr3e-5-bs64-cosine](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/ph8lpetr) (id `ph8lpetr`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run7-v5-lr3e-5-bs64-cosine/error_analysis/` (clusters, confusions, errors_by_severity), built from `run_summary/predictions.csv` with `final-scripts/error_analysis.py`.
- **Data:** `stratified_v5` (SinhaSpeech/sinhala-asr-data), test set 15,860 rows.
- **Hardware:** AMD Instinct MI300X (ROCm 7.14, bf16), PyTorch ROCm nightly.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: cosine
- `per_device_train_batch_size`: 64
- `gradient_accumulation_steps`: 1
- `num_train_epochs`: 5
- `warmup_steps`: 500

## Notes
Identical to run6 except lr_scheduler_type=cosine (linear -> cosine), to isolate the scheduler. Trained in parallel with run6 on the same MI300X.

## Validation per epoch (stratified_v5 validation, 15,763 rows)
| Epoch | WER | CER | Loss |
|---|---|---|---|
| 1 | 30.25% | 8.98% | 0.2045 |
| 2 | 25.43% | 7.60% | 0.1882 |
| 3 | 22.19% | 6.69% | 0.1511 |
| 4 | 20.61% | 6.31% | 0.1365 |
| 5 | 20.35% | 6.23% | 0.1340 |

## Results
- **Val (final):** WER 20.35% / CER 6.23%
- **Test (evaluate_finetuned.py, 15,860 rows):** WER 17.44% / CER 4.72%
- Train runtime: 3.86 h; final train/loss 0.0476

Files: `predictions.csv` (reference,prediction), `training_curves.png`, `lr_schedule.png`, `wandb_config.json`, `wandb_summary.json`.
