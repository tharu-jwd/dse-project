# E007 — Whisper-small wide LoRA, full v4 plus teacher replay

## Status

**Complete. Both gates pass.** Full 220.877-hour v4 split trained to
completion; Sinhala validation shows a real, statistically material
improvement over E006; English retention passes the frozen gate.
This is the final point on the nested Sinhala data-scale curve (E000-E007).

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

## Per-speaker, duration, and length breakdown

Item 2 in [the plan's ranked execution order](../project/plan.md#2-strengthen-evaluation)
requires speaker/duration/transcript-length subgroup reporting and
substitution/deletion/insertion rates on every candidate report; this was
missing from the sections above, so it is added here retroactively against
the same hash-verified predictions
(`f9645dc24424549b262a50382a4db6d1e3a97a19281f0ebd57cffa88bf79708b`).
`sinhala_asr.evaluation.metrics.evaluate_rows` was extended to add
`speaker_id` grouping and data-driven quartile buckets for audio duration
and reference length (quartile boundaries computed from the rows being
scored, not fixed absolute thresholds, so the same labels stay meaningful
on a differently-distributed future benchmark). Regenerated via
`scripts/evaluation/evaluate_predictions.py`; output hashes:
scored.parquet `1e618a4f679a7b2671b8583f169c3cdcef7790434ae53ffb40b9a044227a2ce2`,
summary.json `32c6790c9261deb40506f4516baf1dc5bcb46c9ae5b8a3f63fc546f6e19784a5`.

Canonical word substitution/deletion/insertion rate: 63.51% / 5.76% / 12.44%.
Canonical character substitution/deletion/insertion rate: 11.06% / 11.80% / 3.30%.
Substitutions dominate both, consistent with the earlier spot-check finding
of mostly-correct output with character-level slips rather than deletion-
or insertion-driven degeneracy.

| By speaker | Rows | Canonical WER | Canonical CER |
|---|---:|---:|---:|
| 237ce | 71 | 75.52% | 24.06% |
| ac581 | 63 | 79.72% | 25.63% |
| d7784 | 72 | 89.37% | 28.64% |

Only 3 speakers total in this validation set (the entire speaker-disjoint
held-out pool is 7 speakers split across validation and test -- there is no
larger reserve of unused speakers inside the existing OpenSLR-52-based
corpus). WER varies by up to 14pp by speaker; this is a real spread, not
noise, and is itself part of the case for item 2's independent, more
speaker-diverse benchmark.

| By audio duration (quartile) | Rows | Canonical WER | Canonical CER |
|---|---:|---:|---:|
| q1 (shortest 25%, ≤3.6s) | 52 | 68.11% | 19.94% |
| q2 (25-50%, 3.6-4.5s) | 57 | 78.04% | 21.54% |
| q3 (50-75%, 4.5-6.2s) | 46 | 89.68% | 30.71% |
| q4 (longest 25%, >6.2s) | 51 | 86.83% | 30.16% |

A new finding: error rate rises sharply with clip duration -- shortest-
quartile canonical WER (68.11%) is more than 18pp better than the two
longest quartiles (86.83-89.68%). This was not previously reported for any
experiment in this project despite being required by the evaluation
protocol. It suggests the model's real per-word accuracy on short utterances
is meaningfully better than the aggregate 81.71% WER implies, and that
longer utterances are where most of the error budget concentrates --
plausibly compounding substitution errors or losing alignment over a longer
generated sequence. Worth testing directly once a candidate model is chosen
for further tuning (e.g. does beam search's -5.14pp gain concentrate in the
long-duration quartiles, where greedy has the most room to compound errors).

| By reference length (quartile) | Rows | Canonical WER | Canonical CER |
|---|---:|---:|---:|
| q1 (shortest 25%, ≤4 words) | 95 | 80.69% | 21.04% |
| q2 (25-50%, 4-5 words) | 56 | 84.29% | 23.45% |
| q3 (50-75%, 5-6 words) | 30 | 82.78% | 30.23% |
| q4 (longest 25%, >6 words) | 25 | 78.65% | 36.03% |

Word-count-based length shows a weaker, non-monotonic pattern than audio
duration -- WER stays roughly flat (79-84%) across quartiles while CER rises
in the two longer-transcript quartiles. Duration, not reference word count,
is the stronger predictor of word-level error on this set.

## English retention

Kernel `tharupahan/sinhala-asr-e007-english-evaluation` evaluated the same
hash-verified final adapter against the unchanged 2,620-row LibriSpeech
test-clean benchmark. Runtime: 680.5 seconds on a Tesla T4.

- Benchmark SHA-256:
  `eb1d6f299f5fefde5b66fab450ffbc3b5bf2518ec9e64d3829c050579c6f2906`
- E007 English prediction SHA-256:
  `036041e29fab864cd1af29f9065df005be0fd7286dbc25621bced5843774b563`
  (hash-verified against the kernel's own reported value)

Independently re-scored locally (same 2,620 rows, matched by `sample_id`,
same methodology as every prior experiment):

| Canonical English metric | Untouched | E006 | E007 |
|---|---:|---:|---:|
| WER | 4.2338% | 4.5821% | 4.5971% |
| CER | 1.9240% | 2.0603% | 2.1157% |

Paired bootstrap 95% intervals (2,000 iterations):

- E007 vs. untouched -- WER delta: +0.1766 to +0.5304pp; CER delta: +0.0601
  to +0.2941pp. The point degradation (+0.3633pp) is under the frozen
  0.50-point limit, and the interval's upper bound (+0.5304pp) is under the
  1.00-point limit. **The English-retention gate passes.**
- E007 vs. E006 -- WER delta: -0.1141 to +0.1435pp; CER delta: -0.0216 to
  +0.1392pp. Both intervals include zero: E007 and E006 are statistically
  equivalent on English, the same pattern established at every step since
  E004. Scaling Sinhala exposure from 100 to 220.877 hours did not
  measurably worsen retention under the teacher-replay recipe.

## Decoding-time check: greedy versus beam search

Zero-training follow-up (item 1 in [the plan's ranked execution
order](../project/plan.md#3-ranked-execution-plan)): re-ran the same final
adapter through E007's own original evaluation code path (batched, fp16,
`attention_mask`, `max_new_tokens=64`, same Kaggle T4 platform) with
`num_beams=5` instead of `num_beams=1`, via a new kernel
(`kaggle/e007-decoding-comparison/`) rather than reusing the training
kernel, since this needed to be directly comparable to E007's own reported
number.

A real methodological finding surfaced first: an earlier attempt at this
same comparison, run locally and on Camber, produced individual
predictions that differed from E007's actual recorded output for the same
row, at first suspected to be cross-hardware divergence. Repeating the
check on the identical Kaggle T4 platform still produced a different
per-row prediction for the first validation row -- ruling out
cross-hardware divergence specifically. This project's greedy decoding is
not perfectly deterministic row-by-row on repeat runs (a known property of
unforced-non-deterministic GPU kernels), but the **aggregate** metric over
206 rows is stable: this run's own `num_beams=1` result (81.40% canonical
WER, 26.15% canonical CER) reproduces E007's original reported number
(81.71% WER, 26.15% CER exactly) within a fraction of a point, confirming
the two evaluations are measuring the same underlying model quality
despite individual predictions not being byte-identical. Worth remembering
for any future single-row spot-check: a differing individual prediction is
not on its own evidence of a bug.

Hash-verified predictions (`predictions-beam1.parquet`
`sha256:d6db4ccc6cd2b2f00cd3176a6fe7f01485c5db9ce23d9c475eab9e93f53333cc`,
`predictions-beam5.parquet`
`sha256:778497d674d3ca24031c0d6ace6507db072a035351b86f406d1ddf9b36dc3c34`),
independently re-scored with this project's own metrics:

| Decoding | Canonical WER | Canonical CER |
|---|---:|---:|
| Greedy (`num_beams=1`) | 81.40% | 26.15% |
| Beam search (`num_beams=5`) | **76.26%** | 26.67% |

Beam search gives a real, modest WER improvement (-5.14pp) but CER is
essentially flat, slightly worse (+0.52pp) -- a smaller, more nuanced
effect than an earlier, non-matching-platform attempt at this comparison
suggested (which showed a much larger apparent gap, an artifact of
comparing two runs that had each converged to a different, non-identical
sequence of individual predictions, not a real property of beam search).
Consistent with beam search generally reducing word-level errors (fewer
wrong whole-word choices) without necessarily improving character-level
precision. Zero training cost, composes with any future checkpoint
(rank/LR rerun, tokenizer extension, etc.) as an independent decoding-time
choice on top of whichever model wins.

## Decision rule

Report strict and canonical Sinhala WER/CER with paired bootstrap deltas against
E006, then apply the unchanged English gate: WER point degradation no greater
than 0.50 percentage points and 95% CI upper bound no greater than 1.00 point.

**Both gates pass.** Sinhala improved materially (paired interval excludes
zero) and English retention holds (point degradation and interval upper
bound both within the frozen limits).

## Conclusion: end of the nested data-scale curve

E007 is the last point on the curve begun at E004 (50k rows -> 100 hours ->
220.877 hours). Sinhala error has improved at every single step
(E004 95.87% -> E005 89.57% -> E006 84.48% -> E007 81.71% canonical WER),
with diminishing but still real returns at each stage, while English
retention has passed at every single step. Sinhala remains far above the
under-10% target -- scaling data alone, on this fixed LoRA recipe, will not
close that gap; a materially different lever is needed next. See
[the plan's priority order](../project/plan.md) for what that is: the
rank/LR finding validated in E010 (rank=32, lr~2.3e-4, ready to use in any
future recipe), the tokenizer-extension pilot (E009, fix applied, re-run
pending), and a properly scoped full-parameter pilot, now that this
recipe's own ceiling under LoRA is established with real evidence rather
than projected.
