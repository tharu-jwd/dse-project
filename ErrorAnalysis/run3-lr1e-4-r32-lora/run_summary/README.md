# run3-lr1e-4-r32-lora

- **Canonical name (dse-project finetune_tracker.csv):** `run3-lr1e-4-r32-lora`
- **W&B run:** [run3-lr1e-4-r32-lora](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/5hckpmtf) (id `5hckpmtf`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run3-lr1e-4-r32-lora/error_analysis/` -- originally saved under the inconsistent name `run4_lora`, matched to this run by **hyperparameters** (lr, scheduler, batch size, LoRA config) and renamed to match this run's actual name.
- **Associated eval-only W&B run:** `a56bx95v` (full test-set pass)

## Resume chain (predecessor W&B attempts)
Training was interrupted twice by GPU pod switches (RTX 2000 Ada -> RTX 4090 -> a second RTX 4090), producing 3 crashed W&B runs before the finished one. All logged under the same name `run3-lr1e-4-r32-lora`, which is why hyperparameters (not just name) were needed to pick the right one:
1. `fcbdnl03` -- **crashed** -- 2026-08-26T06:33:32Z (bs=8, grad_accum=4)
2. `ud7yqnge` -- **crashed** -- 2026-08-26T14:35:17Z (bs=32, grad_accum=1 -- a different effective-batch attempt)
3. `4ttoyoqu` -- **crashed** -- 2026-08-26T14:42:13Z (bs=8, grad_accum=4)
4. `5hckpmtf` -- **finished** -- 2026-08-27T03:51:58Z (bs=8, grad_accum=4) — completed epoch 4 (checkpoint-3871 -> checkpoint-11613 -> final); this is the run reported on in this folder.
5. `a56bx95v` -- run3-lr1e-4-r32-lora-**test-eval** -- **finished** -- 2026-08-27T18:17:10Z — separate eval-only run that recorded the full 15,483-row test-set pass against the checkpoint from run 4.

## Hyperparameters (from W&B config)
- `learning_rate`: 0.0001
- `lr_scheduler_type`: cosine
- `per_device_train_batch_size`: 8
- `gradient_accumulation_steps`: 4
- `num_train_epochs`: 4
- `warmup_steps`: 500

## Notes / results (from finetune_tracker.csv)
LoRA (q_proj,k_proj,v_proj,out_proj,fc1,fc2; r=32, alpha=64, dropout=0.05), lr=1e-4 cosine, per_device_bs=8 x grad_accum=4 (eff. 32), 4 epochs. Training interrupted twice by GPU pod switches, resumed each time; converged monotonically (WER 34.48->29.43->26.59->26.01). Val WER 26.01%/CER 7.26%; Test WER 25.99%/CER 7.06%. English forgetting: mild (+2.76pt) -- much smaller than run1/run2 thanks to wider LoRA target-module set. Separate W&B eval-only run 'run3-lr1e-4-r32-lora-test-eval' (a56bx95v) recorded the full 15,483-row test pass on 2026-08-27.

## Final W&B summary metrics
- `eval/wer`: 26.007437101112927
- `eval/cer`: 7.258785677409839
- `eval/loss`: 0.08113355189561844
- `train/loss`: 0.1138
- `train/epoch`: 4

## Contents of this folder
- `training_curves.png` - train/eval loss, eval WER, eval CER vs. step (from W&B history)
- `lr_schedule.png` - learning-rate schedule vs. step
- `wandb_config.json` / `wandb_summary.json` - full W&B run config/summary
- `predictions.csv` - reference/prediction pairs used for error analysis (copied from `ErrorAnalysis/eval_results/`)
- `error_analysis/` - clusters.txt, confusions.txt, errors_by_severity.csv (copied from the existing local error-analysis output)