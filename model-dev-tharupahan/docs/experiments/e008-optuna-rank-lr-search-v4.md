# E008: Automated LoRA rank/learning-rate search (Optuna)

## Status

Complete, but **this is a search, not a controlled experiment** -- its
purpose was to cheaply narrow the rank/LR space, not to produce a trustworthy
standalone result. Its finding (rank=32, lr~2.3e-4 beats this project's
historical default) is not adopted until validated by a real controlled
comparison; see [E010](e010-rank-lr-validation-v4.md).

## Question

Across LoRA rank (8/16/32/64) and learning rate (log-uniform search), which
combination minimizes Sinhala WER fastest, using this project's own E001
recipe as the fixed base and a short 100-step budget as a cheap proxy?

## Frozen recipe (shared across all trials)

- Base: untouched `openai/whisper-small`
- Adaptation: wide LoRA on `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`,
  `fc2`; `lora_alpha = 2 x rank`, dropout 0.05 -- only rank and learning rate
  vary per trial
- Sinhala source: full v4 manifest (`data/versions/v4/manifest.parquet`),
  same 206-row frozen validation set as every other experiment
- Training budget: 100 steps per trial, batch 4, accumulation 4
- Search: Optuna, `scripts/training/optuna_search.py`, minimizing `eval_wer`,
  8 trials requested per job submission

## A real bug found before any trial could be trusted

The first trial run produced `eval_wer~=1.0`/`eval_cer` past 100% on every
attempt: greedy eval-time decoding on an undertrained 100-step checkpoint
degenerated into a repeated-token loop that ran to the generation length
cap, inflating both metrics regardless of the hyperparameters under test.
Found by direct reproduction (loading the checkpoint and inspecting real
generated text, not trusting the aggregate number), fixed in `train.py`
(`no_repeat_ngram_size=3` added to eval-time generation). Full trace:
[the E008 eval-repetition-bug audit](../audits/e008-eval-repetition-bug.md).
Does not affect E005/E006/E007, which train to full length past this
regime.

## A real concurrency bug in how the search ran on Camber

Getting the fixed search to actually run took three Camber job submissions
because of platform-side `PENDING`/auto-`CANCELLED` behavior (see
[the Camber operating notes](../training/camber-cli.md)). An earlier
submission thought abandoned was, as far as could be determined, never
actually stopped remotely (there is no cancel command) and kept writing
into the same shared, stash-synced Optuna SQLite study a later resubmission
also used. Three different snapshots of the resulting study disagreed on
trial count (12 / 15 / 23). Full trace, including why the winning trial is
still trustworthy despite this:
[the E008 rank/LR search results audit](../audits/e008-rank-lr-search.md).

## Result

23 trial directories exist on record (trial count itself uncertain -- see
above); at least 15 are independently confirmed complete via the SQLite
study. All completed trials clustered in a 101-133% WER range at this
100-step budget:

| Trial | Rank | Learning rate | eval_wer |
|---:|---:|---:|---:|
| 6 (best) | 32 | 2.345e-4 | **101.14%** |
| 1 | 16 | 8.015e-5 | 103.72% |
| 4 | 16 | 2.145e-4 | 104.65% |
| 12 | 32 | 1.334e-4 | 104.65% |
| 3 | 64 | 3.585e-5 | 104.55% |
| 13 | 8 | 1.192e-4 | 105.99% |
| 5 | 32 | 3.339e-5 | 108.47% |
| 8 | 16 | 5.899e-5 | 108.88% |
| 11 | 32 | 3.000e-4 | 106.51% |
| 0 | 64 | 1.735e-5 | 111.57% |
| 10 | 64 | 1.721e-5 | 112.60% |
| 9 | 8 | 6.464e-5 | 114.05% |
| 7 | 16 | 3.902e-5 | 115.91% |
| 2 | 16 | 1.792e-5 | 118.08% |

(Full 23-trial list, where recoverable, in the search-results audit above.)

Trial 6's checkpoint was independently re-verified: downloaded and re-run
on 4 real validation clips, producing real, readable, wrong-but-plausible
Sinhala text -- confirms it is not corrupted or a repetition-loop artifact,
though the exact 101.14% figure was not independently re-scored across all
206 rows.

For comparison, E001's own real, fully trusted number at this same
100-step budget with the project's historical default (rank=16, lr=5e-5)
was 114.26% strict WER -- so trial 6's result is directionally consistent
with a real (if modest, and here unvalidated) improvement.

## Conclusion and next step

This search is not evidence on its own -- a 100-step proxy does not
reliably predict which hyperparameters win over a full-length run, and the
search's own bookkeeping was compromised by the concurrency bug above. It
produced a specific, testable hypothesis (rank=32, lr~2.3e-4 over the
historical rank=16, lr=5e-5) and nothing more. See
[E010](e010-rank-lr-validation-v4.md) for the controlled, single-job,
500-step comparison run to actually test it.
