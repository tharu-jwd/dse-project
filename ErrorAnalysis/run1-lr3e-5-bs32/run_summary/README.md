# run1-lr3e-5-bs32

- **Canonical name (dse-project finetune_tracker.csv):** `run1-lr3e-5-bs32`
- **W&B run:** [run1-lr3e-5-bs32](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/peti3e5t) (id `peti3e5t`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run1-lr3e-5-bs32/error_analysis/` -- originally saved under the inconsistent name `run1_full`, matched to this run by **hyperparameters** (lr, scheduler, batch size, LoRA config) and renamed to match this run's actual name.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 8
- `gradient_accumulation_steps`: 4
- `num_train_epochs`: 4
- `warmup_steps`: 500

## Notes / results (from finetune_tracker.csv)
Full fine-tune, lr=3e-5 linear, per_device_bs=8 x grad_accum=4 (eff. 32), 4 epochs, warmup 500. Val WER 19.96%/CER 4.17%; Test WER 17.08%/CER 3.49%. English forgetting: severe (+76.59pt WER).

## Final W&B summary metrics
- `eval/wer`: 19.96063876157392
- `eval/cer`: 4.16636224154307
- `eval/loss`: 0.10654214769601822
- `train/loss`: 0.0347
- `train/epoch`: 4

## Contents of this folder
- `training_curves.png` - train/eval loss, eval WER, eval CER vs. step (from W&B history)
- `lr_schedule.png` - learning-rate schedule vs. step
- `wandb_config.json` / `wandb_summary.json` - full W&B run config/summary
- `predictions.csv` - reference/prediction pairs used for error analysis (copied from `ErrorAnalysis/eval_results/`)
- `error_analysis/` - clusters.txt, confusions.txt, errors_by_severity.csv (copied from the existing local error-analysis output)