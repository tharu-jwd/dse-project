# Sinhala ASR Rebuild Plan

This file is the durable source of truth for the clean Sinhala ASR pipeline.
Update it whenever a decision changes or a phase is completed. Do not rely on
chat history or undocumented notebook state.

## Objective

Build a reproducible preprocessing, fine-tuning, evaluation, and error-analysis
pipeline for Sinhala ASR. Start each controlled baseline from an official
pretrained multilingual Whisper checkpoint, initially
`openai/whisper-small`, rather than from a team fine-tuned checkpoint or random
weights.

In this document, "official baseline" means an untouched OpenAI checkpoint; it
does not mean the checkpoint named `openai/whisper-base`. Use Whisper-base only
for inexpensive pipeline smoke tests. Whisper-small is the primary training and
comparison model. Consider Whisper-medium only after a small-model recipe wins
on the frozen validation protocol and passes a new cost review.

The first acceptance target is less than 10% strict WER and less than 10%
strict CER on a frozen, leakage-checked test set. Canonical metrics are reported
alongside strict metrics but cannot replace them. Standalone English retention
is desirable rather than a hard acceptance condition; code-switched English in
Sinhala samples remains part of the primary task and is never silently removed.

Existing checkpoints and results are historical comparison baselines only. The
superseded `model-development/` tree was audited, its relevant evidence was
captured in the historical audit, and it was removed from this branch. It
remains recoverable from Git history and the team branches; the clean pipeline
has no runtime dependency on it.

See [the historical audit](../audits/historical-audit.md) for the evidence-graded run
history, contradictions, evaluation policy, and experiment rationale.

## Non-negotiable rules

1. Do not rent a GPU until data validation, metrics, tests, smoke tests, and
   checkpoint resume work locally.
2. Do not use the test set for model or decoding selection.
3. Do not compare runs trained on different dataset or normalization versions.
4. Every run must record its Git commit, full resolved configuration, dataset
   fingerprint, split fingerprint, seed, package versions, hardware, duration,
   planned cost, and actual cost.
5. Training and evaluation paths must be configurable; no machine-specific or
   hard-coded cloud paths.
6. Training must be resumable. From E002 onward, Colab jobs do not mount Google
   Drive: checkpoints are downloaded into the experiment's local artifact
   directory and verified before a cloud instance is terminated. Follow the
   heartbeat, checkpoint-completion, attempt-log, bounded-retry, and
   deterministic-resume procedure in
   [the Colab CLI policy](../training/colab-cli.md) (historical, E000-E002) or
   [the Kaggle operations policy](../training/kaggle-cli.md) (current, E003
   onward), whichever platform applies.
7. Generated datasets, checkpoints, predictions, and reports are not committed
   unless they are deliberately selected compact reference artifacts.

Historical metrics, run contradictions, and inherited operational lessons live
only in [the historical audit](../audits/historical-audit.md); do not duplicate
them here. Keep implementation
under `src/sinhala_asr/`, entry points under `scripts/`, tests under `tests/`,
and generated data, reports, runs, and checkpoints in ignored directories.

## Phase 1: dataset audit and manifest

Reconstruct the corpus from independently downloadable upstream sources first:
official OpenSLR-52 plus the public YouTube Sinhala and BizBrains datasets.
Download Linga only for provenance comparison because it appears to overlap
OpenSLR substantially and must not be counted as independent speech. Preserve
each upstream snapshot unchanged with its source URL, revision, license, file
checksums, and download date. Use the team's GCS `finalData` later as a
comparison artifact to recover and verify useful corrections—not as assumed
ground truth or as a prerequisite for beginning the audit.

Build a deterministic manifest containing at least:

- Stable sample ID
- Source dataset
- Source record ID
- Speaker ID when available
- Audio content hash
- Transcript hash
- Duration, sample rate, channels, and encoding
- Original transcript
- Canonical transcript
- Code-switch flag
- Split assignment
- Validation flags and exclusion reason

Produce reports for corrupt or missing audio, silence, clipping, duration
outliers, Unicode anomalies, exact and near duplicates, repeated transcripts,
speaker leakage, audio leakage, and source/speaker/split distributions.

Dataset adequacy is measured primarily in verified speech hours, speakers, and
conditions—not row count. The audit must report total and retained hours,
duration quantiles, speaker coverage where identities exist, source/domain
coverage, Sinhala-only/code-switched counts, and estimated transcript error
rates. The historical claim of approximately 154,828 rows is not accepted as a
training-capacity measurement until source snapshots are fingerprinted and
their retained speech hours are calculated.

Perform a reviewed, stratified listening audit before freezing the data. The
local review UI must play audio, show original/canonical text and audit flags,
support keyboard-driven correct/edit/bad-audio/mismatch/duplicate/uncertain
decisions, save progress continuously, and export a versioned adjudication
table without modifying raw sources.

Review at least 100 rows from each applicable category: random OpenSLR, random
collection sources, automatically flagged anomalies, duplicate/near-duplicate
candidates, and code-switched speech. A row may satisfy more than one category.
Use this sample to estimate error rates, not as the sole training corpus.

Exit with failure when invariants are violated. Never silently discard a row;
write its reason to an exclusions manifest.

Regenerate speaker-disjoint splits when reliable speaker or recording-group
identifiers can be recovered. Also retain a collection-domain robustness test,
a standalone English-retention set, and explicit Sinhala-only and code-switched
evaluation slices. If speaker identity is unavailable, document that limitation
and group by the strongest defensible recording/session identifier instead.

Before training, the native-Sinhala reviewer must lock a gold evaluation set of
approximately 1,000–2,000 diverse clips, divided into validation and test. Both
parts are excluded from training. Validation may select recipes and checkpoints;
test remains unopened until a candidate is frozen. Store corrections as overlays
and publish immutable dataset versions (`v1`, `v2`, and so on), never by
overwriting upstream data.

## Phase 2: Sinhala text policy

Preserve three representations:

- `text_original`: immutable source transcript
- `text_canonical`: consistent training target
- `text_metric`: explicitly normalized scoring representation

The policy must document Unicode NFC, whitespace, punctuation, numbers, dates,
abbreviations, Sinhala compounds and particles, colloquial/formal variants,
English words, and transliteration. Normalization must be deterministic,
versioned, and unit-tested with Sinhala examples.

Never hide recognition failures by over-normalizing. Report strict and
canonical metrics together.

## Phase 3: evaluation and error analysis

Every prediction row must retain manifest metadata and include reference,
prediction, strict/canonical scores, and error labels.

Required aggregate views:

- Strict WER and CER
- Canonical WER and CER
- Sinhala-only and code-switched WER
- WER by source, speaker, duration, and transcript length
- Named-entity, number, and domain-term performance when labeled
- Substitution, deletion, and insertion rates
- Empty output, truncation, repetition, and hallucination indicators
- Bootstrap confidence intervals for headline metrics

Required sample-level error taxonomy:

- Punctuation only
- Whitespace or compound segmentation
- Accepted spelling variant
- Colloquial/formal mismatch
- Sinhala grapheme confusion
- Acoustic substitution
- Deletion
- Insertion
- Named entity
- Number
- Code-switch
- Truncation
- Repetition or hallucination
- Suspected bad reference
- Suspected bad audio

Generate machine-readable tables and a human-readable Markdown or HTML report.

## Phase 4: training

Begin with one reproducible `openai/whisper-small` baseline. Support LoRA/DoRA
training behind the same configuration schema. Track strict/canonical
validation metrics, loss, learning rate, gradient norm, throughput, peak memory,
checkpoint identity, and wall time.

Initial controlled experiments after evaluating the untouched model:

1. Wide-target LoRA or DoRA from official `openai/whisper-small`, selecting a
   conservative learning rate through short pilots.
2. Bilingual replay using the adapter recipe plus a fixed English rehearsal set
   and the available Sinhala-English code-switched samples.
3. Adapter target-width and learning-rate comparison on the same frozen data.
4. Low-learning-rate continuation of the historical 17% checkpoint only if its
   exact artifact, optimizer state, and dataset identity can be recovered --
   and only after re-measuring its accuracy on the current frozen,
   speaker-disjoint v4 evaluation set. [The historical audit](../audits/historical-audit.md)
   has since confirmed the original 17%/25.99% historical numbers were measured
   on a split with pervasive speaker leakage (a plain random row-level split
   with no speaker column at all), so neither is a trustworthy target or
   comparison point until re-measured; this option is deprioritized relative to
   item 3 accordingly.
5. Augmentation ablations only after error analysis shows the matching need.
6. Whisper-medium only after the winning small-model recipe and budget review.

### Next-step priority order after E007

The nested data-scale curve (E004-E007) is now fully measured, not just
projected: canonical WER improved at every step (95.87% -> 89.57% -> 84.48%
-> **81.71%** at E007, the complete 220.877-hour split), with English
retention passing every time, and diminishing but still real returns at
each stage. Further scale is not expected to close the remaining gap to the
under-10% target -- the fitted curve already implied this before E007 ran,
and E007's real result confirms it rather than overturning it. [The
near-homophone error analysis](../audits/e006-near-homophone-error-analysis.md)
shows most of the remaining error is genuine, broader recognition failure,
not a small fixable confusion set. Rank the next levers by expected benefit
per unit of time/risk, not by raw expected benefit alone:

**Foundational findings (complete, feeding into the forward-looking order
below):**

- Cheap, no-GPU diagnostic: how Whisper's tokenizer encodes the specific
  Sinhala near-homophone character pairs identified in the error analysis.
  Done: `openai/whisper-small`'s tokenizer has **zero dedicated Sinhala
  tokens** at all (every one of the 128 Sinhala Unicode codepoints is two
  raw UTF-8 byte-level fallback tokens; none of the 51,865 vocabulary
  entries decode to clean standalone Sinhala). See
  [the near-homophone analysis](../audits/e006-near-homophone-error-analysis.md#tokenizer-check-whisper-small-has-zero-dedicated-sinhala-vocabulary)
  for the full finding, including why ල/ළ specifically -- the single most
  frequent confusion found -- gets no help from either the acoustic signal
  or the token representation.
- Adapter rank/learning-rate search (E008) via
  `scripts/training/optuna_search.py` -- **complete**. First attempt
  produced an invalid trial (greedy eval-time decoding on an undertrained
  100-step checkpoint degenerated into a repeated-token loop that pinned
  `eval_wer~=1.0`; found by direct reproduction, fixed in `train.py`,
  see [the E008 eval-repetition-bug audit](../audits/e008-eval-repetition-bug.md)).
  Moved to Camber once fixed; hit a real concurrency bug there (an earlier
  "abandoned" job kept running remotely and wrote into the same shared
  Optuna study a later resubmission also used) -- winning trial
  independently re-verified despite it. **Result: rank=32,
  learning_rate~2.3e-4 beats this project's historical default (rank=16,
  lr=5e-5)** -- 101.14% vs E001's real 114.26% strict WER at the same 100
  steps. See [the E008 experiment report](../experiments/e008-optuna-rank-lr-search-v4.md).
- Controlled validation of E008's finding (E010) -- **complete, finding
  validated and adopted**. A clean, single-job, no-concurrency-risk
  500-step comparison (5x E008's proxy scale) confirms rank=32,
  lr~2.345e-4 beats the historical default at every single eval
  checkpoint, not just the final one: eval_wer 97.42% vs 102.89% (-5.47pp),
  eval_cer 33.13% vs 43.71% (-10.58pp). See
  [the E010 report](../experiments/e010-rank-lr-validation-v4.md).
- Tokenizer vocabulary extension (E009) -- **build done, first pilot found
  unstable, fix applied, re-run pending.** Two independent structural
  findings (phoneme identity + zero-Sinhala-vocabulary) motivated this fix
  for the project's single worst-offending confusion pair, with real
  external precedent for both safety and expected magnitude (roughly -0.7
  percentage points WER in the closest published study). Built
  `scripts/training/extend_tokenizer.py`: 250 new tokens from the real v4
  corpus, 57.1% tokens-per-word reduction (10.214 -> 4.386), above the
  external precedent's 30-61% range. First pilot came back with WER
  *rising* over training (169% -> 999% between step 50 and 100) and the
  model outputting actual Khmer script on two of four sampled clips.
  Root cause confirmed structurally: this project's LoRA `target_modules`
  doesn't include Whisper's tied embedding/output-projection layer, so the
  250 new token rows stayed frozen at random initialization for the whole
  run -- exactly the embedding-freezing risk flagged before building
  anything, now confirmed for this project's LoRA recipe specifically (the
  external paper's "no instability" result came from full fine-tuning,
  where nothing is frozen, so it never tested this failure mode). Fix
  applied and committed: `modules_to_save=["embed_tokens", "proj_out"]`
  plus `ensure_weight_tying=True`, verified directly (every other
  experiment's LoRA config unchanged, same trainable param count). See
  [the E009 experiment report](../experiments/e009-tokenizer-extension-pilot-v4.md).
  Second pilot run, with the fix, not yet done.

**Forward-looking priority order, ranked by expected improvement per unit
of training time -- decided explicitly to save full-parameter fine-tuning
for last: by the time that (expensive, effectively one-shot) run happens,
every cheaper lever should already be tried and folded into the recipe
going in, not discovered afterward:**

1. Re-run E009's tokenizer pilot with the fix (~15-20 minutes, Camber).
   Nearly free, and the one open unknown blocking items 3 and 4 below --
   needs answering before tokenizer extension can be counted as one of the
   validated tricks to carry forward.
2. Full-scale rerun of E007's recipe with the validated rank/LR (rank=32,
   learning_rate~2.3e-4) -- same cost as E007 itself (~11h Kaggle,
   two-phase). Doesn't need to wait on item 1; already justified by E010's
   evidence. Highest-confidence big-ticket item, because unlike the other
   levers it isn't a guess -- E010 demonstrated a real, consistent 5.5 to
   10.6 percentage-point gain at every checkpoint of a 500-step comparison.
   E007 itself never used this recipe (it used the E001-era rank=16,
   lr=5e-5 default), so this also doubles as the first real controlled
   comparison against E007 at full scale, which E010's 500-step run
   doesn't substitute for.
3. Full-scale tokenizer-extension experiment (~11h Kaggle) -- only if item
   1 succeeds. Same cost class as item 2, smaller expected payoff by the
   one real precedent available (~-0.7pp WER), though this project's own
   57.1% token-compression ratio beat that paper's, so it's plausible the
   real gain here exceeds -0.7pp.
4. Combine every validated trick into one recipe -- rank/LR (item 2) plus
   tokenizer extension if it holds (item 3) plus NEFTune
   (`TrainConfig.neftune_noise_alpha`, already wired into both the local
   and Kaggle/Colab training paths via an optional job-config field, cheap
   to try, untested at Sinhala-adapter scale here). This is the actual
   "exhausted the options" step, not items 2 or 3 individually -- the real
   predecessor to full-parameter training, not a substitute for it.
5. A human-in-the-loop correction batch targeted specifically at rows
   containing the near-homophone confusions identified in the error
   analysis, rather than a random sample -- higher signal per reviewed
   row. Spends reviewer time, not GPU time, so it can run in parallel with
   any of the above rather than competing for the same budget; fold any
   resulting corrections into the training data before the full-parameter
   run, not after.
6. Full-parameter fine-tuning -- **deliberately last.** Only start once
   items 1-5 have reported back, so the recipe going in is whatever
   combination of rank/LR, tokenizer extension, NEFTune, and corrected data
   actually proved out, plus a real number for how much of the remaining
   gap is architecture-limited (LoRA's ceiling) versus something a full
   update could fix -- not a guess made before the cheaper options were
   exhausted. Potentially the largest remaining lever: two independent
   external pipelines show the same qualitative pattern (full fine-tune
   reaching materially lower WER than this project's LoRA ceiling, even
   under far less data/rigor) -- see
   [the SPEAK-ASR/ASR-Finetune review](../audits/asr-finetune-external-review.md)
   and
   [the whisper-based-sinhala-asr review](../audits/whisper-based-sinhala-asr-external-review.md).
   Neither is individually trustworthy (one has confirmed split leakage,
   the other's split is simply unverifiable), and the historical
   project's own leakage-tainted 17% number is a reason to check, not
   evidence this will work -- but three independent codebases landing on
   the same direction is a reason to take the pilot seriously once it's
   this recipe's turn. A bounded feasibility/VRAM/step-time scoping pilot
   (this project's own Gate B discipline) belongs immediately before the
   real run, not run early just because it's cheap -- there's no reason to
   spend even a small amount of time on it before it's actually needed.

Before full-data recipe comparisons, measure a nested data learning curve using
approximately 10, 25, 50, and 100 verified speech hours plus the full retained
dataset. Build every larger subset as a superset of the smaller one, balanced by
speaker and source where metadata permits. Use identical bounded pilot settings
and validation evaluation. Advance full runs only when the curve shows that
additional data is useful; diagnose label noise, domain mismatch, tokenization,
or capacity when it plateaus.

Change one experimental factor at a time. Use validation data and early stopping
for selection; evaluate the test set only after freezing a candidate.

### Human-in-the-loop improvement cycle

Do not require manual verification of the full training corpus. Train initially
on automatically high-confidence data plus reviewed corrections. After each
baseline, rank training candidates for review using high loss, low confidence,
checkpoint disagreement, repeated error patterns, transcript/audio mismatch
signals, and underrepresented speakers or domains. The reviewer then corrects a
meaningful batch—normally 300–1,000 samples, a completed error category, or at
least one verified hour—before another GPU run.

For controlled comparisons, restart every candidate from the same official
Whisper-small checkpoint. For final staged adaptation, continuation from the
winning checkpoint is allowed at a lower learning rate, but mix prior training
data with new corrections to reduce overfitting and forgetting. Periodically
retrain from the official checkpoint on the complete latest dataset version to
detect bias accumulated through repeated continuation. Never fine-tune on the
gold validation or test rows.

## Phase 5: GPU and cloud cost gates

Initial compute allowance: five included Camber GPU hours plus at most USD 10
of paid compute. Free hours are still budgeted and measured. The GPU model,
VRAM, framework compatibility, persistence, and observed throughput must be
recorded before estimating how many complete runs fit this allowance.

Status: CLI installed, authenticated, and Gate A/B-equivalent smoke-tested
(2026-09-06) -- a live job confirmed a real NVIDIA L4 (23034MiB) reachable
via `--gpu --size xsmall`, and confirmed the `base` engine's default image
does *not* ship PyTorch despite its own description claiming otherwise (any
real job needs its own `pip install` step). ~5 minutes of the 5-hour budget
spent on that one verification job; the CPU-only probe that found the
missing-torch/no-conda details cost nothing against the GPU budget. See
[the Camber operating notes](../training/camber-cli.md) for the full
verified CLI surface, node-size constraints, and environment findings.
No training job has run there yet -- holding it for a concrete GPU-bound
task (most likely the tokenizer-extension pilot, item 3 above) rather than
spending the budget on anything already covered locally or on Kaggle.

### Gate A: free/local checks

- Unit and integration tests pass
- Dataset audit passes
- Metrics pass fixed fixtures
- CPU smoke test completes
- A checkpoint can be saved, loaded, and resumed
- A small prediction report is generated successfully

### Gate B: capped GPU smoke test

- Use a fixed step limit and small data subset
- Verify mixed precision and peak VRAM
- Verify data-loader throughput and GPU utilization
- Verify checkpoint upload and recovery
- Deliberately interrupt and resume once
- Verify automatic cleanup and shutdown behavior

### Gate C: measured pilot

Run 5-10% of the training data or 500-1,000 steps. Record samples/second,
evaluation time, checkpoint time, peak VRAM, and total billed time. Estimate:

```text
planned cost = hourly GPU price × estimated hours × 1.25 safety margin
```

Do not start a full run without an approved maximum cost and stop condition.
The first pilot must calculate the projected total cost; if it exceeds the
remaining allowance, the job must not advance automatically.

### Gate D: bounded full experiments

- Run one baseline first
- Advance only the best one or two short pilots
- Run one full winner
- Attempt a larger model only when the measured expected benefit justifies cost

Keep immutable source snapshots and canonical outputs on the local machine with
checksums and a separate backup. Upload only the required frozen dataset version
to a training provider, and download run records, predictions, and important
checkpoints before terminating it. Do not use a stopped GPU volume as long-term
storage. Record the instance type and displayed hourly price at run start, and
terminate compute automatically after success or error.

For Colab specifically (historical, E000-E002), follow
[the CLI isolation policy](../training/colab-cli.md): use only
`/content/sinhala-asr-job` remotely, never mount Drive, and keep durable
per-experiment artifacts under `reports/experiments/eNNN-...` locally. For
Kaggle (current, E003 onward), follow
[the Kaggle operations policy](../training/kaggle-cli.md): verify dataset/
kernel-source anchor filenames are unique before every push, poll with
`kernels output` on a multi-minute interval rather than `kernels status`, and
split a run into checkpoint-resumed stages ahead of time if it is projected
close to the platform's session ceiling.

## Completion criteria

The project is ready for paid full training only when:

- A frozen, fingerprinted, leakage-checked dataset exists
- Sinhala normalization is documented and tested
- Untouched Whisper baseline results are recorded
- Detailed error reports work end to end
- Training smoke and resume tests pass
- Pilot throughput and cost are measured
- The full run config and maximum spend are approved

The project is complete when a selected checkpoint has reproducible strict and
canonical test results, subgroup/error analysis, English-retention results when
required, deployment latency measurements, a model card, and a documented
actual cloud cost.

## Progress

- [x] Historical run and error-profile review
- [x] Clean-project scope and architecture recorded
- [x] Package scaffold and development tooling
- [x] Dataset manifest and audit tooling
- [x] Conservative Sinhala normalization v1 and tests
- [x] Download and fingerprint independently available upstream datasets
- [x] Run source and cross-source audits on actual audio/transcript data
- [x] Build the local adjudication UI and self-contained review queues
- [x] Review and lock audio-verified validation/test sets (dataset v4)
- [x] Generate deterministic speaker-disjoint v1 candidate splits
- [x] Audit full-corpus boundary silence and clipping without modifying sources
- [x] Test boundary trimming; reject it for the baseline after manual review and local A/B
- [x] Implement strict/canonical metrics, error labels, subgroups, and confidence intervals
- [x] Implement local Whisper training, prediction, and reporting paths
- [x] Enforce configuration-based cloud cost and test-set access gates
- [ ] Evaluation and detailed error reports
- [x] Configuration-driven training
- [x] Local smoke and checkpoint-resume tests
- [x] Capped 100-step free-Colab wide-LoRA pilot and measured throughput
- [x] Untouched Whisper-small v4 validation baseline
- [x] Freeze the full LibriSpeech test-clean English-retention benchmark and
  run the untouched Whisper-small English baseline
- [x] First controlled preprocessing experiment (trimmed versus original audio)
- [x] Controlled transcript-label refinement A/B (Sinhala-only and Latin-only;
  automatic text-only refinement rejected; dataset v3 remains unchanged)
- [x] Audio-verify 295 disputed evaluation references and 100 unchanged controls;
  freeze 392 usable references as v4 and hold out 1,604 unheard rows
- [x] Benchmark Bedrock-accessible transcript refiners against 293 audio-verified
  targets; Sonnet 4.6 is safest but does not reproduce the earlier ChatGPT pass
- [x] Model-training controlled experiments (E000-E007 complete: the nested
  Sinhala data-scale curve through the full 182,665-row/220.88-hour split,
  trained in two checkpoint-resumed Kaggle stages, with material Sinhala
  gains and passing English retention at every step; canonical WER 81.71%,
  CER 26.15% at E007, still far above the under-10% target -- data scale
  alone on this fixed LoRA recipe will not close that gap)
- [ ] Final test, deployment benchmark, and model card
