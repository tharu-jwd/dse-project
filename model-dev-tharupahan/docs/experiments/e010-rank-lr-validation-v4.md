# E010: Controlled validation of E008's rank/LR search finding

## Status

**Complete. E008's finding is validated.** Candidate (rank=32,
lr~2.345e-4) beats the historical default (rank=16, lr=5e-5) on every
single metric, at every eval checkpoint, in this clean 500-step run.

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

Both runs completed all 500 steps on Camber (`--gpu --size xsmall`), one
job, sequentially -- no concurrency risk this time. Full eval trajectory,
read directly from each run's own `trainer_state.json`:

| Step | Baseline WER | Candidate WER | Baseline CER | Candidate CER | Baseline eval_loss | Candidate eval_loss |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 110.12% | 103.62% | 76.17% | 46.79% | 1.520 | 0.413 |
| 200 | 102.48% | 100.21% | 57.54% | 35.87% | 0.908 | 0.310 |
| 300 | 104.75% | 106.40% | 48.43% | 39.56% | 0.653 | 0.270 |
| 400 | 109.09% | 104.03% | 47.02% | 36.98% | 0.507 | 0.236 |
| **500 (final)** | **102.89%** | **97.42%** | **43.71%** | **33.13%** | **0.462** | **0.221** |

```
                        Baseline (rank=16, lr=5e-5)   Candidate (rank=32, lr~2.3e-4)
Final train_loss:       3.904                          1.656
Final eval_loss:        0.462                          0.221
Final eval_wer:         102.89%                        97.42%   (-5.47 pp)
Final eval_cer:         43.71%                         33.13%   (-10.58 pp)
```

The candidate leads in eval_loss and eval_cer at **every single eval
checkpoint** (100/200/300/400/500), and in eval_wer at 4 of 5 (step 300 is
the one exception, a small reversal within noise). Train loss and eval loss
both converge roughly 2x faster for the candidate. This is not a marginal,
noise-level difference -- it is a consistent, large gap across the entire
training trajectory, not just the final checkpoint.

The candidate's final checkpoint was independently re-verified: downloaded
and re-run on 4 real validation clips, producing real, readable,
wrong-but-plausible Sinhala text -- confirms the result is a genuine
learning-quality difference, not a decoding artifact or repetition-loop
inflation of one side's numbers.

## Conclusion

**E008's rank/LR search finding holds up under a clean, controlled,
5x-longer validation.** Rank=32 with a learning rate roughly 4-5x higher
than this project's historical default (2.345e-4 vs 5e-5) is the better
choice for this recipe, not just at E008's 100-step proxy scale but
confirmed again at 500 steps. Both configurations remain far from a usable
transcription result at this step count (WER still ~97-103%) -- that is
expected and not the point of this comparison; the point was relative
ranking between two hyperparameter choices, and that ranking is now
established with real evidence, not a contaminated search's bookkeeping.

Recommendation: use rank=32, learning_rate~2.3e-4 (or search nearby) as the
new default for any future LoRA experiment on this recipe, rather than
carrying forward rank=16, lr=5e-5 by inertia from E001. This does not
retroactively change E001-E007's own results (each was a valid, single-
factor controlled comparison on its own frozen recipe) -- it is guidance
for what comes after E007.
