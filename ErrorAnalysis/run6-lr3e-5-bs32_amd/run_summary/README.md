# run6-lr3e-5-bs32_amd

- **Canonical name (dse-project finetune_tracker.csv):** `run6-lr3e-5-bs32_amd`
- **W&B run:** [run6-lr3e-5-bs32_amd](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/rcb54cuc) (id `rcb54cuc`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run6-lr3e-5-bs32_amd/error_analysis/` -- originally saved under the inconsistent name `run2_amd_lora`, matched to this run by **hyperparameters** (lr, scheduler, batch size, LoRA config) and renamed to match this run's actual name.

## Hyperparameters (from W&B config)
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 32
- `gradient_accumulation_steps`: 2
- `num_train_epochs`: 4
- `warmup_steps`: 500

## Notes / results (from finetune_tracker.csv)
LoRA (q_proj,v_proj; r=32, alpha=64, dropout=0.05) on AMD MI300X-class pod, lr=5e-5 (despite the local eval-folder name implying 3e-5) linear, per_device_bs=32 x grad_accum=2 (eff. 64), 4 epochs. Val WER 23.81%/CER 6.46%; Test WER 21.01%/CER 5.73%. English forgetting: severe.

## Final W&B summary metrics
- `eval/wer`: 23.81366656121481
- `eval/cer`: 6.460315109536449
- `eval/loss`: 0.12085644900798798
- `train/loss`: 0.0309
- `train/epoch`: 4

## Contents of this folder
- `training_curves.png` - train/eval loss, eval WER, eval CER vs. step (from W&B history)
- `lr_schedule.png` - learning-rate schedule vs. step
- `wandb_config.json` / `wandb_summary.json` - full W&B run config/summary
- `predictions.csv` - reference/prediction pairs used for error analysis (copied from `ErrorAnalysis/eval_results/`)
- `error_analysis/` - clusters.txt, confusions.txt, errors_by_severity.csv (copied from the existing local error-analysis output)