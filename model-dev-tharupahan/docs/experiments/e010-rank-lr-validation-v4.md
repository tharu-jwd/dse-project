# E010: Controlled validation of E008's rank/LR search finding

## Status

Running (Camber, job `25201`). Not yet complete -- this document will be
updated with real results once both runs finish.

## Question

E008's Optuna search suggested rank=32, lr~2.345e-4 beats this project's
historical default (rank=16, lr=5e-5) -- but that search was an explicitly
cheap 100-step proxy with its own bookkeeping compromised by a real
Camber-side concurrency bug (see [E008](e008-optuna-rank-lr-search-v4.md)).
Does the finding hold up in a clean, single-job, no-concurrency-risk,
longer (500-step, matching E002's scale) controlled comparison?

## Frozen recipe (both runs identical except rank/learning rate)

- Base: untouched `openai/whisper-small`
- Adaptation: wide LoRA on `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`,
  `fc2`, dropout 0.05
- Sinhala source: full v4 manifest, same 206-row frozen validation set
- Training budget: 500 steps, batch 4, accumulation 4 -- one order of
  magnitude past E008's 100-step proxy regime
- **Baseline** (`configs/training/experiments/e010-rank-lr-validation-baseline-v4.json`):
  rank=16, alpha=32, lr=5e-5 -- this project's historical default, matching
  E001/E002
- **Candidate** (`configs/training/experiments/e010-rank-lr-validation-candidate-v4.json`):
  rank=32, alpha=64, lr=2.345e-4 -- E008's best trial's exact hyperparameters

Both runs execute sequentially inside one Camber job (not two concurrent
submissions), specifically to avoid the class of bug documented in E008's
report.

## Result

Pending.

## Conclusion

Pending.
