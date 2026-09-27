# run2-lr3e-5-bs32_lora

- **Canonical name (dse-project finetune_tracker.csv):** `run2-lr3e-5-bs32_lora`
- **W&B run:** [run2-lr3e-5-bs32_lora](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/xiv7fh6f) (id `xiv7fh6f`, state `crashed`)
- **Error-analysis folder:** `ErrorAnalysis/run2-lr3e-5-bs32_lora/error_analysis/` -- originally saved under the inconsistent name `run3_lora`, matched to this run by **hyperparameters** (lr, scheduler, batch size, LoRA config) and renamed to match this run's actual name.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: cosine
- `per_device_train_batch_size`: 8
- `gradient_accumulation_steps`: 4
- `num_train_epochs`: 4
- `warmup_steps`: 500

## Notes / results (from finetune_tracker.csv)
LoRA (q_proj,v_proj; r=32, alpha=64, dropout=0.05), lr=3e-5 cosine, per_device_bs=8 x grad_accum=4 (eff. 32). Crashed/interrupted at ~epoch 2.5/4 -- Val WER 61.63%/CER 18.81% are mid-training, non-converged values, not comparable to finished runs. This is the run labelled 'best-epoch2'/outlier in the local error-analysis writeup.

## Final W&B summary metrics
- `eval/wer`: 61.62738584836023
- `eval/cer`: 18.807459642966997
- `eval/loss`: 0.34415891766548157
- `train/loss`: 0.2934
- `train/epoch`: 2.5252212103597493

## Contents of this folder
- `training_curves.png` - train/eval loss, eval WER, eval CER vs. step (from W&B history)
- `lr_schedule.png` - learning-rate schedule vs. step
- `wandb_config.json` / `wandb_summary.json` - full W&B run config/summary
- `predictions.csv` - reference/prediction pairs used for error analysis (copied from `ErrorAnalysis/eval_results/`)
- `error_analysis/` - clusters.txt, confusions.txt, errors_by_severity.csv (copied from the existing local error-analysis output)