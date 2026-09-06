# E007 — Whisper-small wide LoRA, full v4 plus teacher replay

## Status

**Training complete (both phases). Sinhala validation complete and
independently re-scored -- material, statistically significant improvement
over E006. English-retention evaluation running (kernel pushed, not yet
back).**

Phase A: kernel version 1 started 2026-09-06 12:17 Asia/Colombo, finished
19:16 (Kaggle T4, ~6.9h). Reached the intended stop at step 4,077 with
`train_loss=1.917` -- healthy. Checkpoint hash-verified locally and staged
as a Kaggle dataset for Phase B to resume from.

Phase B: kernel version 2 (version 1 crashed immediately on a real,
now-documented Kaggle platform behavior -- dataset ingestion recursively
auto-extracts any uploaded archive, so the checkpoint tarball never
survived upload intact; see
[the Kaggle operations policy](../training/kaggle-cli.md) and
`install_resume()` in `scripts/training/run_e007_kaggle.py` for the fix)
ran to the full target: **step 6,342, one complete effective epoch over
the entire 220.877-hour v4 training split.** Completed 2026-09-07 ~01:11
Asia/Colombo. Final training loss in the last logged steps: ~1.0-1.1,
consistent with continued healthy convergence (compare E006's 2.318 mean
loss over its whole run -- not a like-for-like number, but directionally
consistent).

## Question

Does scaling the successful E006 recipe from 100 hours to the complete v4
training split continue to reduce Sinhala error while retaining the untouched
model's English capability?

## Frozen design

- Base model and adapter: Whisper-small, wide LoRA rank 16, unchanged from E006.
- Sinhala source: all 182,665 v4 training rows (220.877 hours), ordered by
  `sample_id`; selection fingerprint
  `78399cdea0f1b98a17aebe08db5fc8611eaae5be58f881fedf1c17fea3c10cbb`.
- English retention replay: 20,296 occurrences from the same 1,111 frozen
  LibriSpeech clips and untouched-model teacher targets used by E004–E006.
- Total: 202,961 training occurrences and 6,342 optimizer steps, approximately
  one effective epoch with the established global effective batch of 32.
- Checkpoints: every 453 steps. Phase A intentionally stops at step 4,077;
  Phase B must resume the exact saved optimizer, scheduler, scaler, RNG, and
  trainer state and continue to step 6,342.
- Evaluation remains the frozen 206-row Sinhala validation set followed by the
  frozen 2,620-row LibriSpeech English-retention benchmark.

The two-phase boundary keeps each Kaggle job below its 12-hour session ceiling.
It is one continuous training run, not a fresh adapter continuation. Phase B
will not be created until Phase A's exact checkpoint archive hash is known.

## Memory and failure controls

The 183 Sinhala audio shards are verified individually and converted one at a
time, avoiding simultaneous Arrow and Python copies of the full 12.7 GB corpus.
Only the 1,111 unique English audio rows are stored; replay occurrences reuse
the same immutable audio-byte objects in memory. Every durable checkpoint is
archived with hashes, and Phase B refuses to resume unless the step-4,077
archive hash and trainer state match the frozen handoff.

## Preparation verification

- Semantic source revision: `b5fe7a21855849508f6a8a9c0c29416031a82304`.
- Full Sinhala shard set: 183 shards, 12,689,696,104 bytes, combined SHA-256
  `ecd5bbffee8a644fd2db83385799838ca055cc832335aaa90de00ca7db35e6f9`.
- Local Kaggle staging preflight re-hashed every Sinhala shard and the frozen
  English/validation inputs. It produced 185 training shards containing
  183,776 stored rows; deterministic replay expansion yields the expected
  202,961 occurrences.
- The private Kaggle orchestration runtime was published with independently
  indexed copies of the generic runner and E007 orchestrator. The full audio
  input dataset reached Kaggle's `ready` state with all 186 expected server
  files present and all indexed payload sizes matching local sources.

## Execution

Phase A is running as private Kaggle kernel
`tharupahan/sinhala-asr-e007-full-v4-teacher-replay-phase-a`, version 1. Its
expected durable handoff is `checkpoint-004077.tar.gz`. No Phase B kernel will
be published until that archive is downloaded and independently hashed.

## Training and Sinhala validation

Phase B's kernel produced `sinhala-validation-predictions.parquet` (206
real per-row predictions). Hash-verified against the kernel's own reported
values before use:

- Final adapter SHA-256:
  `4c92dee2fbdc7bc59ef70e7e17654967b47cf8b535956f67a09c2c1458acc865`
- Checkpoint-index SHA-256:
  `9d7fe37f6f490bc2a3c7678e9689697ec2b678bd5dd0699d6e20c7dad7f0cfb7`
- Resume checkpoint (step 6,342) SHA-256:
  `b8c9bcae7c8fbda685268f64d437e5c38cdea512d608b0a2352160342377022c`
- Sinhala prediction SHA-256:
  `f9645dc24424549b262a50382a4db6d1e3a97a19281f0ebd57cffa88bf79708b`

Independently re-scored locally with this project's own `score_pair`/
`aggregate` (`strict_normalize`/`metric_normalize`), not just read off a
Kaggle-side summary:

| Sinhala validation metric | E006 | E007 |
|---|---:|---:|
| Strict WER | 85.33% | 82.95% |
| Strict CER | 29.82% | 27.20% |
| Canonical WER | 84.48% | 81.71% |
| Canonical CER | 28.74% | 26.15% |

Paired bootstrap 95% intervals (E007 minus E006, canonical, 2,000
iterations, same 206 rows matched by `sample_id`, same reference text
confirmed identical between the two prediction sets before scoring):

- WER delta: -5.48 to -0.21 percentage points -- excludes zero.
- CER delta: -3.81 to -1.45 percentage points -- excludes zero.

Both intervals exclude zero: the complete 220.877-hour split produces a
real, statistically material improvement over E006's 100-hour tier, not
sampling noise. The improvement is smaller than E005-to-E006's own gain,
continuing the diminishing-returns pattern already visible across the
curve, and Sinhala remains far above the under-10% target.

A spot check of the raw predictions (not just the aggregate) confirms real,
mostly-correct Sinhala output rather than any decoding degeneracy --
e.g. `ඒත් අපිට ලැබුරු උතර පට්ටපල්වරයු` against reference
`ඒත් අපිට ලැබුණු උත්තර පට්ට පල් බොරු`, close but for a handful of
character-level errors, a qualitatively different regime from the ~80-100%+
gibberish seen in this project's various short-step-budget diagnostic runs
(E008/E009/E010).

## English retention

Kernel `tharupahan/sinhala-asr-e007-english-evaluation` pushed and running
against the same hash-verified final adapter. This section will be updated
with real, independently-scored results once it returns -- no conclusion is
recorded until both language evaluations are in.

## Decision rule

Report strict and canonical Sinhala WER/CER with paired bootstrap deltas against
E006, then apply the unchanged English gate: WER point degradation no greater
than 0.50 percentage points and 95% CI upper bound no greater than 1.00 point.
No conclusion is recorded until both language evaluations finish.
