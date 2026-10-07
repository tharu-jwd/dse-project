# run9-v6-lr3e-5-bs64

- **W&B run:** [run9-v6-lr3e-5-bs64](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/905z0tif) (id `905z0tif`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run9-v6-lr3e-5-bs64/error_analysis/` (clusters, confusions, errors_by_severity), built from `run_summary/predictions.csv` with `final-scripts/error_analysis.py`.
- **Data:** `stratified_v6` (SinhaSpeech/sinhala-asr-data), test set 15,860 rows -- the first run trained on v6, which canonicalises spelling/ZWJ/register/typo issues found and resolved during this session (see tracker Notes for `run10-v6-lr3e-5-bs64`).
- **Hardware:** AMD Instinct MI300X (ROCm 6.2 nightly torch, bf16), at 134.199.196.125.
- **Note on naming:** this folder and the pod's own output-dir/W&B run are named `run9-...`, predating a tracker-only rename to `run10-v6-lr3e-5-bs64` (done to avoid colliding with a pre-existing, differently-configured `run9` already logged in the tracker before this session). The pod itself was not renamed.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 64
- `gradient_accumulation_steps`: 1
- `num_train_epochs`: 5
- `warmup_steps`: 500

Identical to `run6-v5-lr3e-5-bs64` in every hyperparameter -- the only variable changed is the dataset (`stratified_v6` instead of `stratified_v5`), to isolate the effect of the data cleanup alone.

## Validation per epoch (stratified_v6 validation, 15,763 rows)
| Epoch | WER | CER | Loss | vs run6-v5 (v5 data) |
|---|---|---|---|---|
| 1 | 29.61% | 8.88% | 0.1746 | run6-v5: 30.49% (v6 better by 0.88) |
| 2 | 25.36% | 7.56% | 0.1854 | run6-v5: 25.14% (v6 worse by 0.22) |
| 3 | 22.39% | 6.70% | 0.1490 | run6-v5: 22.68% (v6 better by 0.29) |
| 4 | 20.85% | 6.42% | 0.1397 | run6-v5: 21.13% (v6 better by 0.28) |
| 5 | 20.15% | 6.23% | 0.1330 | run6-v5: 20.27% (v6 better by 0.12) |

Validation WER beat run6-v5 at 4 of 5 epochs, monotonic improvement throughout.

## Results
- **Val (final):** WER 20.15% / CER 6.23%
- **Test (evaluate_finetuned.py, 15,860 rows):** WER 17.17% / CER 4.71%
- Train runtime: 4.83 h; final train/loss 0.0498

**Test-set result did not confirm the validation-set improvement.** Despite winning validation on 4 of 5 epochs, the full test-set WER (17.17%) came out 0.02 points *worse* than run6-v5's 17.15% (CER also slightly worse: 4.71% vs 4.62%). This is effectively a statistical tie, not a clear win -- consistent with this session's own prior estimate that the v6 canonicalisation (touching ~1.9% of training rows) was projected to move WER by only ~0.24-0.34 points on its own, well within the noise of a single training run. The v6 data cleanup is still a legitimate, low-risk fix (it removes genuine training-data contradictions), but this run does not provide clean evidence that it improves this model's real-world accuracy over v5 at this configuration.

Files: `predictions.csv` (reference,prediction), `training_curves.png`, `lr_schedule.png`, `wandb_config.json`, `wandb_summary.json`.
