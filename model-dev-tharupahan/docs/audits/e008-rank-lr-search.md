# E008: Automated LoRA rank/learning-rate search (Optuna)

Evidence grade: **Verified artifact** for the winning trial (downloaded and
directly re-ran); **Interpretation** for the total trial count and the
non-winning trials' individual integrity -- see the concurrency finding
below.

## What this is

`scripts/training/optuna_search.py` searches LoRA `rank` (8/16/32/64) and
`learning_rate` (log-uniform) against this project's own established E001
recipe (100 steps, v4 manifest, teacher-replay not included -- a fast proxy
search, not a full controlled experiment), minimizing `eval_wer`. See
[the training guide](../training/training.md#automated-hyperparameter-search-and-neftune)
for the design. Blocked early by a real bug (undertrained checkpoints
degenerating into a runaway repeated-token loop during eval, inflating
`eval_wer`/`eval_cer` past any real signal) -- found, fixed, and documented
separately in
[the E008 eval-repetition-bug audit](e008-eval-repetition-bug.md) before
this search ever produced a trustworthy result.

## A real concurrency finding, not a bug in the search logic itself

Getting this search to actually run on Camber took three job submissions
(`25196`, `25199`, `25200`) because of two separate platform issues
documented in
[the Camber operating notes](../training/camber-cli.md#a-gpu-job-can-sit-pending-for-a-long-time-then-auto-cancel):
`25196` sat `PENDING` indefinitely and was abandoned (its poller killed,
but **the remote job itself was never explicitly stopped** -- there is no
cancel command); `25199` sat `PENDING` for ~96 minutes then Camber
auto-cancelled it. `25200` finally ran and printed, in its own live log,
`Completed 23 total trial(s). Best trial: #6, WER=1.0114`.

But the study's own synced state is inconsistent across three different
snapshots pulled afterward:

- `search-summary.json` on stash: 12 trials, #11 still `RUNNING`.
- `study.db` (the live SQLite file) on stash: 15 trials, #14 still
  `RUNNING`.
- `25200`'s own printed log: 23 trials, all complete.
- Trial directories that physically exist on stash: `trial-000` through
  `trial-022` -- 23, matching the log's count. `trial-022`'s `train.log` is
  **0 bytes**, unlike a normal completed trial's ~15KB log.

The most likely explanation: `25196`, thought abandoned, was never actually
stopped remotely and kept running against the same persistent
`sqlite:///runs/e008-rank-lr-search/study.db` in the same shared stash
path -- so when `25200` was later submitted against that same path, it
picked up an already-in-progress study and appended its own trials on top,
with SQLite (not built for multiple independent machines writing to one
file through an async stash sync layer) losing some writes along the way.
This is a genuine, newly-found risk of this project's current
Camber-plus-persistent-Optuna-study setup: **there is no way to confirm a
seemingly-stuck Camber job has actually stopped**, and a study.db shared
across overlapping job instances is not safe.

## What's actually trustworthy here

The winning trial's own result was independently re-verified, not just
read off the summary: downloaded `trial-006`'s `checkpoint-100` adapter
directly and re-ran generation on four real validation clips (same
direct-reproduction method used for the E008 repetition-bug and E009
frozen-embedding findings). Output was real, readable, plausible Sinhala
text -- wrong in the way an undertrained 100-step checkpoint should be
wrong, not degenerate:

```
REF: ඒත් අපිට ලැබුණු උත්තර පට්ට පල් බොරු
HYP: එහිතවි දෙබුණු උතර පටට පළල් බොරු
```

**Best trial (verified): #6 -- `lora_rank=32`, `learning_rate≈2.345e-4`,
`eval_wer≈1.0114` (101.14%).** For comparison, E001's own real, fully
trusted number at this same 100-step budget with the project's previous
default (rank=16, lr=5e-5) was 114.26% strict WER -- so this is a real,
modest improvement, consistent with what a rank/LR search should find, not
a breakthrough.

Across the (at least 15, up to 23) trials that did complete, results
clustered tightly in the 101-133% range regardless of hyperparameters, with
a visible pattern: rank 32 and 64 trials generally outperformed rank 8 and
16, and learning rates around 1e-4 to 3e-4 outperformed both the project's
historical default (5e-5, too low at this step budget) and the low end of
the search space (~1.7e-5, far too low). No trial's raw output showed the
repetition-loop degeneracy the eval-repetition-bug fix was built to
prevent -- confirms that fix is holding across a real spread of
hyperparameters, not just the one checkpoint it was originally diagnosed
on.

## Practical takeaway

`rank=32` with a learning rate meaningfully higher than this project's
historical default (2-5x higher, in the ~1e-4-3e-4 range) is the
better-supported choice for any future short-step-budget LoRA experiment on
this recipe, not `rank=16, lr=5e-5`. This does not retroactively change
E001-E007 (those were controlled comparisons on their own frozen recipe,
not searches), but it is real evidence for how any *next* full-scale
experiment (e.g. the tokenizer-extension pilot, once its own frozen-
embedding fix is re-run) should pick its rank/LR rather than reusing the
E001 default by inertia.

## Process fix needed before any future Optuna-on-Camber run

Do not resubmit a new job against a `--path` holding an in-progress Optuna
study without first confirming the previous job is genuinely stopped -- and
since there is no cancel command, the only reliable way to confirm that
today is to wait for the job to reach a real terminal state (`COMPLETED`/
`FAILED`/`CANCELLED`) via `camber job get`, not to assume "I gave up
polling it" means "it stopped."
