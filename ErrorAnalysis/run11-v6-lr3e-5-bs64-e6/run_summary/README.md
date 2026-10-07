# run11-v6-lr3e-5-bs64-e6

- **W&B run:** [run11-v6-lr3e-5-bs64-e6](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/1n2t4gpf) (id `1n2t4gpf`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run11-v6-lr3e-5-bs64-e6/error_analysis/` (clusters, confusions, errors_by_severity), built from `run_summary/predictions.csv` with `final-scripts/error_analysis.py`.
- **Data:** `stratified_v6` (SinhaSpeech/sinhala-asr-data), test set 15,860 rows -- same dataset as `run10-v6-lr3e-5-bs64`.
- **Hardware:** AMD Instinct MI300X (ROCm 6.2 nightly torch, bf16), at 134.199.196.125.
- **Purpose:** run10 trained 5 epochs on v6 and came out essentially tied with run6-v5 on the test set despite winning validation. This run repeats the exact same config for 6 epochs, to test whether the extra epoch (not the data cleanup itself) is what was missing.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 64
- `gradient_accumulation_steps`: 1
- `num_train_epochs`: 6
- `warmup_steps`: 500

Identical to `run10-v6-lr3e-5-bs64` and `run6-v5-lr3e-5-bs64` except for one more training epoch.

## Validation per epoch (stratified_v6 validation, 15,763 rows)
| Epoch | WER | CER | Loss | vs run10 (v6, 5ep) | vs run6-v5 (v5, 5ep) |
|---|---|---|---|---|---|
| 1 | 29.44% | 8.75% | 0.2197 | run10: 29.61% (better by 0.17) | run6-v5: 30.49% (better by 1.05) |
| 2 | 25.11% | 7.50% | 0.1771 | run10: 25.36% (better by 0.25) | run6-v5: 25.14% (better by 0.03) |
| 3 | 22.60% | 6.74% | 0.1606 | run10: 22.39% (worse by 0.21) | run6-v5: 22.68% (better by 0.08) |
| 4 | 21.20% | 6.48% | 0.1327 | run10: 20.85% (worse by 0.35) | run6-v5: 21.13% (worse by 0.07) |
| 5 | 20.04% | 6.20% | 0.1284 | run10: 20.15% (better by 0.11) | run6-v5: 20.27% (better by 0.23) |
| 6 | 19.61% | 6.17% | 0.1316 | run10 stopped at 5 epochs | run6-v5 stopped at 5 epochs |

Validation improvement was uneven epoch-to-epoch relative to run10 (trading places through epochs 3-4), but the 6th epoch clearly pushed past both 5-epoch runs. Eval loss ticked up slightly from epoch 5 to 6 (0.1284 -> 0.1316) while eval WER/CER kept improving -- a mild divergence worth noting but not a clear overfitting signal, since train loss (~0.114 average over epoch 6) kept falling smoothly the whole run with no plateau.

## Results
- **Val (final, epoch 6):** WER 19.61% / CER 6.17%
- **Test (evaluate_finetuned.py, 15,860 rows):** WER 16.38% / CER 4.43%
- Train runtime: ~5.75 h

**This is the best result in the project so far.** Test WER improved 0.79 points over run10 (17.17% -> 16.38%) and 0.77 points over run6-v5 (17.15% -> 16.38%); test CER improved 0.28 points over run10 (4.71% -> 4.43%) and 0.19 points over run6-v5 (4.62% -> 4.43%). Unlike run10, which only tied run6-v5 despite the v6 data cleanup, the extra training epoch here produced a clear, unambiguous win on the held-out test set -- suggesting the model had not yet converged at 5 epochs, and the v6 data cleanup's benefit may only become visible once the model is trained further.

Files: `predictions.csv` (reference,prediction), `training_curves.png`, `lr_schedule.png`, `wandb_config.json`, `wandb_summary.json`.
