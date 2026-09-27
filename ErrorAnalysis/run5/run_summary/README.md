# run5-v4-lr3e-5-bs64-v3-resume

- **Canonical name (dse-project finetune_tracker.csv):** `run5-v4-lr3e-5-bs64-v3-resume`
- **W&B run:** [run7-v4-lr3e-5-bs64-v3-resume](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/qww5akos) (id `qww5akos`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run5-v4-lr3e-5-bs64-v3-resume/error_analysis/` -- originally saved under the inconsistent name `run5_full`, matched to this run by **hyperparameters** (lr, scheduler, batch size, LoRA config) and renamed to match this run's actual name.

## Resume chain (predecessor W&B attempts)
This run is the last link in a 4-attempt chain, all logged as separate W&B runs under the (misleading) name `run7-v4-lr3e-5-bs64*`. Only the final attempt finished and is the one used for evaluation/error analysis above.
1. `8bpxy2xm` -- run7-v4-lr3e-5-bs64 -- **failed** -- 2026-09-08T11:36:15Z (bs=64, grad_accum=1)
2. `rot9ket4` -- run7-v4-lr3e-5-bs64 -- **failed** -- 2026-09-08T11:37:59Z (bs=16, grad_accum=4)
3. `f9no0ylf` -- run7-v4-lr3e-5-bs64-v2 -- **crashed** -- 2026-09-08T13:33:42Z (bs=8, grad_accum=8)
4. `qww5akos` -- run7-v4-lr3e-5-bs64-v3-resume -- **finished** -- 2026-09-09T13:11:51Z (bs=8, grad_accum=8) — resumed from checkpoint-5778 of attempt 3 after a transformers/torch.load (CVE-2025-32434) version incompatibility; this is the run reported on in this folder.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 8
- `gradient_accumulation_steps`: 8
- `num_train_epochs`: 4
- `warmup_steps`: 500

## Notes / results (from finetune_tracker.csv)
Full fine-tune, lr=3e-5 linear, per_device_bs=8 x grad_accum=8 (eff. 64), 4 epochs, warmup 500. Resumed from checkpoint-5778 after a transformers/torch.load (CVE-2025-32434) incompatibility forced a pod switch and a pinned-version fix. load_best_model_at_end=True (metric=wer). Val WER 22.41%/CER 6.65%; Test WER 19.06%/CER 4.90% (test notably better than val, unlike other runs). English forgetting check skipped for this run (not evaluated).

## Final W&B summary metrics
- `eval/wer`: 22.41171954964176
- `eval/cer`: 6.646411785307181
- `eval/loss`: 0.14322230219841003
- `train/loss`: 0.0598
- `train/epoch`: 3.9999350691513538

## Contents of this folder
- `training_curves.png` - train/eval loss, eval WER, eval CER vs. step (from W&B history)
- `lr_schedule.png` - learning-rate schedule vs. step
- `wandb_config.json` / `wandb_summary.json` - full W&B run config/summary
- `predictions.csv` - reference/prediction pairs used for error analysis (copied from `ErrorAnalysis/eval_results/`)
- `error_analysis/` - clusters.txt, confusions.txt, errors_by_severity.csv (copied from the existing local error-analysis output)