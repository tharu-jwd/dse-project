# run6-v5-lr3e-5-bs64

- **W&B run:** [run6-v5-lr3e-5-bs64](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/of5yw35a) (id `of5yw35a`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run6-v5-lr3e-5-bs64/error_analysis/` (clusters, confusions, errors_by_severity), built from `run_summary/predictions.csv` with `final-scripts/error_analysis.py`.
- **Data:** `stratified_v5` (SinhaSpeech/sinhala-asr-data), test set 15,860 rows.
- **Hardware:** AMD Instinct MI300X (ROCm 7.14, bf16), PyTorch ROCm nightly.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 64
- `gradient_accumulation_steps`: 1
- `num_train_epochs`: 5
- `warmup_steps`: 500

## Notes
Full fine-tune, lr 3e-5 linear, per-device bs 64 x grad_accum 1 (eff. 64), 5 epochs, warmup 500, stratified_v5, on an AMD MI300X pod (bf16). Trained in parallel with run7 on the same GPU.

## Validation per epoch (stratified_v5 validation, 15,763 rows)
| Epoch | WER | CER | Loss |
|---|---|---|---|
| 1 | 30.49% | 9.01% | 0.2120 |
| 2 | 25.14% | 7.48% | 0.1831 |
| 3 | 22.68% | 6.77% | 0.1633 |
| 4 | 21.13% | 6.38% | 0.1371 |
| 5 | 20.27% | 6.21% | 0.1336 |

## Results
- **Val (final):** WER 20.27% / CER 6.21%
- **Test (evaluate_finetuned.py, 15,860 rows):** WER 17.15% / CER 4.62%
- Train runtime: 4.00 h; final train/loss 0.0493

Files: `predictions.csv` (reference,prediction), `training_curves.png`, `lr_schedule.png`, `wandb_config.json`, `wandb_summary.json`.
