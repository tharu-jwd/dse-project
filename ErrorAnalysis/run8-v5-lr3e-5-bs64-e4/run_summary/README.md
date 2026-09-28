# run8-v5-lr3e-5-bs64-e4

- **W&B run:** [run8-v5-lr3e-5-bs64-e4](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/zl7shcql) (id `zl7shcql`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run8-v5-lr3e-5-bs64-e4/error_analysis/` (clusters, confusions, errors_by_severity), built from `run_summary/predictions.csv` with `final-scripts/error_analysis.py`.
- **Data:** `stratified_v5` (Yohan2003/whisper-sl-data), test set 15,860 rows.
- **Hardware:** AMD Instinct MI300X (ROCm 7.14, bf16), PyTorch ROCm nightly.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 64
- `gradient_accumulation_steps`: 1
- `num_train_epochs`: 4
- `warmup_steps`: 500

## Notes
Identical to run6 except num_train_epochs=4 (5 -> 4), to measure what the fifth epoch is worth.

## Validation per epoch (stratified_v5 validation, 15,763 rows)
| Epoch | WER | CER | Loss |
|---|---|---|---|
| 1 | 30.15% | 8.82% | 0.2222 |
| 2 | 24.91% | 7.49% | 0.1732 |
| 3 | 22.54% | 6.83% | 0.1608 |
| 4 | 21.12% | 6.43% | 0.1506 |

## Results
- **Val (final):** WER 21.12% / CER 6.43%
- **Test (evaluate_finetuned.py, 15,860 rows):** WER 17.90% / CER 4.76%
- Train runtime: 2.32 h; final train/loss 0.0593

Files: `predictions.csv` (reference,prediction), `training_curves.png`, `lr_schedule.png`, `wandb_config.json`, `wandb_summary.json`.
