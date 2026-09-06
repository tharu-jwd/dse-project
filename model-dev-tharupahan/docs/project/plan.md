# Sinhala ASR Development Plan

Last updated: 2026-09-07

This is the canonical project plan. Detailed experiment narratives belong in
docs/experiments/, audits in docs/audits/, and operational instructions in
docs/training/. This file records only current decisions, execution order,
gates, and completion criteria.

## 1. Objective and success criteria

Build a reproducible Sinhala ASR system with:

- Strict WER below 10%.
- Strict CER below 10%.
- Canonical WER and CER reported alongside strict metrics.
- Evaluation on a frozen, fingerprinted, leakage-checked test set.
- Standalone English retention measured, although it is not a hard acceptance
  condition.
- Sinhala-English code-switching retained as part of the primary task.

The target must not be reached through speaker leakage, training-corpus overlap,
or aggressive normalization. Published CER is not equivalent to WER, and
neither counts as independent when evaluated audio may have appeared during
model pretraining.

## 2. Current state

### Data

- Dataset v4 is the current controlled dataset.
- Training: 182,665 OpenSLR-52 rows, 220.877 hours, 471 speakers.
- Validation: 206 rows, 17.6 minutes, 3 speakers.
- Test: 186 rows, 16.2 minutes, 4 speakers.
- Train, validation, and test speakers do not overlap.
- Raw sources are immutable; corrections are versioned overlays.
- Automatic GPT/Bedrock transcript rewriting was not adopted as a general data
  policy. Only audio-verified corrections are trusted for evaluation labels.

The validation split is honest but too small and speaker-narrow to estimate
broad generalization. It remains frozen for continuity while a genuinely
independent and more speaker-diverse benchmark is constructed.

See the [dataset specification](../data/dataset.md),
[text policy](../data/text-policy.md), [audio audit](../data/audio-audit.md),
and [review provenance](../data/review-provenance.md).

### Best completed controlled model

E007 adapted openai/whisper-small using wide LoRA rank 16,
teacher-behavior English replay, and all 220.877 training hours:

| Metric | E006 | E007 |
|---|---:|---:|
| Canonical Sinhala WER | 84.48% | 81.71% |
| Canonical Sinhala CER | 28.74% | 26.15% |
| English WER | 4.5821% | 4.5971% |

The Sinhala improvement is statistically material and English retention
passes. Scaling this fixed LoRA recipe nevertheless shows diminishing returns
and cannot plausibly close the remaining gap alone.

E007 varies substantially across the three validation speakers:

| Speaker | Rows | Canonical WER | Canonical CER |
|---|---:|---:|---:|
| 237ce | 71 | 75.52% | 24.06% |
| ac581 | 63 | 79.72% | 25.63% |
| d7784 | 72 | 89.37% | 28.64% |

See the [E007 report](../experiments/e007-whisper-small-wide-lora-r16-full-v4-teacher-replay.md).

### Established technical findings

1. Whisper-small has no dedicated Sinhala tokens; Sinhala uses byte-level
   fallback tokens.
2. E009 added 250 corpus-derived Sinhala tokens and reduced tokens per word
   from 10.214 to 4.386 (-57.1%).
3. E009's first result is invalid evidence against tokenizer extension because
   LoRA froze its new tied embedding/output rows. The corrected configuration
   trains embed_tokens and proj_out with weight tying; its rerun is pending.
4. E010 confirmed rank 32 and learning rate approximately 2.345e-4 beat the
   historical rank-16/5e-5 setting at 500 steps. This finding is retained but
   does not automatically justify an expensive full-scale Whisper rerun.
5. Historical team results near 17% WER used a random row-level split with
   speaker leakage and are not valid comparisons on the current protocol.
6. External/team checkpoints are comparison baselines, not trusted training
   starting points.

See the [tokenizer audit](../audits/e006-near-homophone-error-analysis.md),
[E009 report](../experiments/e009-tokenizer-extension-pilot-v4.md),
[E010 report](../experiments/e010-rank-lr-validation-v4.md), and
[historical audit](../audits/historical-audit.md).

## 3. Ranked execution plan

This order is ranked by expected benefit per unit of training time. Change it
only when a recorded result changes the evidence.

### 1. Complete the E007 decoding comparison

**Complete.** Beam search (`num_beams=5`) gives a modest, real WER
improvement over greedy (76.26% vs 81.40% canonical WER) with CER
essentially flat (26.67% vs 26.15%) -- confirmed on the same Kaggle T4
platform and code path as E007's own original evaluation, not a different
platform's non-matching baseline. As expected, does not close the WER gap
to target; adopt beam search as the default decoding choice going forward
regardless of which model wins the bake-off below, since it is free and
composes with any checkpoint. See
[the E007 report](../experiments/e007-whisper-small-wide-lora-r16-full-v4-teacher-replay.md#decoding-time-check-greedy-versus-beam-search).

### 2. Strengthen evaluation

- Preserve the 206-row validation set for historical comparability. Done,
  unchanged.
- **Done for E007.** Per-speaker/duration/transcript-length subgroup
  reporting and substitution/deletion/insertion rates, previously required
  by section 4 but never actually produced for any experiment report, are
  now built into `sinhala_asr.evaluation.metrics.evaluate_rows` (data-driven
  quartile buckets, not fixed thresholds) and added to
  [the E007 report](../experiments/e007-whisper-small-wide-lora-r16-full-v4-teacher-replay.md#per-speaker-duration-and-length-breakdown).
  New finding: canonical WER rises from 68.11% (shortest-duration quartile)
  to 86.83-89.68% (two longest quartiles) -- error concentrates in longer
  clips, not evenly spread. Still owed: backfilling this same breakdown for
  E004-E006, and re-running it whenever a new candidate is scored.
- Add a seen-speaker holdout only as a diagnostic for separating adaptation
  failure from unseen-speaker generalization failure. **Not started, and
  cannot be applied retroactively to E007** -- E007 already trained on 100%
  of the v4 train split, so there is no already-trained-on-but-withheld
  sample to mine after the fact. Carving one out means a future training
  run trains on slightly less than "the complete split" in order to reserve,
  e.g., ~300-500 rows spread across many train speakers as a same-speaker/
  held-back-utterance eval set. That changes what "full training data" means
  for every subsequent experiment (items 6, 7, 9 below) -- worth adopting,
  but a decision for a human to confirm before it's baked into a config,
  not something to decide silently.
- Build an independent benchmark with more speakers and recording
  conditions. **Checked, and the existing OpenSLR-52-based corpus cannot
  supply this**: the entire speaker-disjoint held-out pool is exactly 7
  speakers (`heldout_unreviewed`/`heldout_unused` in `data/versions/v4/manifest.parquet`
  are the same 7 speakers already split across validation and test, not
  additional ones) -- confirmed by direct inspection, not assumed. The
  only candidate source with real speaker diversity already indexed in this
  repo is the SPEAK-ASR YouTube corpus
  (`reports/dataset-audit/youtube-upstream/manifest.parquet`: 4,037 rows,
  9.1 hours, 34 distinct uploaders as a speaker/condition proxy -- BizBrains
  and the Lingalingeswaran JSON are each single-uploader and don't help
  here). Its licensing is explicitly "Unresolved; private audit only until
  clarified" per [the source policy](../data/dataset.md#source-policy).
  Using it even eval-only, kept private and never redistributed, is a
  licensing/compliance call outside what this plan has decided -- flagged
  for a human decision, not resolved unilaterally.
- Fingerprint it and exclude it from adaptation and model selection. Applies
  once the above is decided.
- Keep the existing test set unopened until a candidate is frozen. Still
  holding -- test set untouched.

### 3. Run a zero-training model-family bake-off

Evaluate candidates using identical audio, references, normalization, and
metrics:

1. Meta Omnilingual ASR CTC 300M v2.
2. Omnilingual CTC 1B v2 if memory permits.
3. Omnilingual LLM-ASR 1B and 7B as compute permits.
4. E007 Whisper-small.
5. Recoverable team/external checkpoints as controls.

Meta's [official results](https://raw.githubusercontent.com/facebookresearch/omnilingual-asr/refs/heads/main/per_language_results_table_7B_llm_asr.csv)
report Sinhala at 7.2% CER with 225.4 training hours, and its
[official repository](https://github.com/facebookresearch/omnilingual-asr)
provides inference and fine-tuning recipes. Because 225.4 hours closely matches
this OpenSLR corpus, assume possible overlap until disproved. Do not claim
independent performance without the new benchmark.

### 4. Adapt Omnilingual CTC 300M

Proceed only if its bake-off result is competitive:

- Use a direct Sinhala grapheme inventory.
- Run a fixed-step bounded pilot before full training.
- Compare untouched and adapted checkpoints.
- Measure throughput, memory, checkpoint recovery, validation change, and cost.
- Advance to CTC 1B only after 300M demonstrates honest transfer.

CTC is prioritized because it avoids Whisper's Sinhala byte-token bottleneck
and supports efficient language-model-assisted decoding.

### 5. Add Sinhala language-model decoding

Apply character/subword or word-level KenLM beam decoding to the strongest CTC
candidate. Consider a neural LM only if KenLM establishes a measurable gain.

- Tune LM weight and word-insertion penalty on validation only.
- Report acoustic-only and LM-assisted scores together.
- Inspect spelling, compound, and word-boundary changes.
- Do not use an LM to conceal acoustic hallucinations or reference errors.

This ranks early because E007's 26.15% CER versus 81.71% WER indicates that
character, spelling, and segmentation errors amplify the word metric.

### 6. Re-run the corrected E009 tokenizer pilot

**Stopped, as this item's own stop condition specifies.** Ran on Camber
(job 25203) with the fix applied. The structural fix is confirmed correct
-- the previous wrong-script (Khmer) failure is gone -- but the corrected
pilot still collapses to a different degenerate output (a short repeated
loop of bare Sinhala vowel-sign marks, not real words) and does not beat
E010 on any metric (canonical WER/CER ~100%/86-91% at step 100). Verified
directly, not just from the reported eval numbers: a local spot-check
loaded the real downloaded adapter and generated actual predictions for 5
validation clips, confirming the degenerate pattern in the raw text
itself. See
[the E009 report](../experiments/e009-tokenizer-extension-pilot-v4.md#corrected-re-run).
Not escalated to a longer run -- this item's own condition says to stop
here, and the likely cause (250 newly-added embedding rows need far more
than ~1,600 effective training examples to adapt) would need a materially
larger, costlier budget to even test fairly.

### 7. Compare full Whisper adaptation with LoRA

Run a bounded, identical-data, identical-step comparison:

- Full-parameter Whisper-small with a conservative learning rate.
- Rank-32 LoRA at approximately 2.345e-4.
- Same seed, rows, decoder, checkpoints, and validation.
- English-retention evaluation for both.

Advance full adaptation only if it substantially breaks the LoRA ceiling.
Protect English through measured replay/regularization rather than assuming
full fine-tuning necessarily destroys it.

### 8. Test alternative multilingual CTC families if needed

If Omnilingual is unavailable or leaves a material gap, evaluate MMS Sinhala
support and XLS-R. Use bounded output-head/encoder adaptation before any full
run. Do not train large models merely to create a leaderboard.

### 9. Run optimized Whisper LoRA at full scale only if competitive

E010 established rank-32/higher-LR as better at 500 steps. Run an E007-scale
version only if the model-family bake-off shows Whisper remains competitive
enough to justify approximately another 11-hour job.

### 10. Run evidence-triggered ablations

Test one factor at a time after the winning architecture and error profile are
known:

- DoRA or adaptive LoRA.
- SpecAugment, speed/pitch perturbation, or noise when errors concentrate by
  speaker, channel, or noise.
- NEFTune where supported.
- Targeted human correction for recurring high-impact label/error categories.
- Continued self-supervised pretraining or pseudo-labeling only when useful
  additional unlabeled Sinhala audio exists.

### 11. Consider larger models

Consider Whisper-medium or larger full runs only after the bake-off, bounded
full-parameter pilot, and cost review. A larger Whisper retains the same
Sinhala-tokenization limitation; parameter count alone is not justification.

## 4. Evaluation protocol

Every prediction row must retain sample ID, speaker/source metadata, reference,
prediction, strict/canonical edit counts, and error labels.

Every candidate report must include:

- Strict and canonical WER/CER.
- Sinhala-only and code-switched results.
- Results by speaker, source, duration, and transcript length.
- Substitution, deletion, and insertion rates.
- Whitespace/compound, grapheme, spelling-variant, named-entity, number,
  truncation, repetition, hallucination, bad-reference, and bad-audio analyses.
- Paired bootstrap confidence intervals against the relevant control.
- Representative improved and regressed samples.
- An explanation of why metrics changed and how the evidence determines the
  next experiment.

The test set is evaluated only after the candidate, normalization policy, and
decoder are frozen. See the [evaluation specification](../evaluation/evaluation.md).

## 5. Experiment discipline

1. Change one experimental factor at a time.
2. Use the same dataset and normalization version for controlled comparisons.
3. Record Git revision, resolved configuration, data/split fingerprints, seed,
   packages, hardware, duration, planned/actual cost, and artifact hashes.
4. Make training resumable and validate restoration before scaling.
5. Preserve per-row predictions and failure logs, including unsuccessful runs.
6. Never silently discard samples; record exclusion reasons.
7. Do not commit generated data/checkpoints unless deliberately selected as
   compact reference artifacts.
8. Do not continue an external checkpoint as the final model unless a future
   plan revision explicitly justifies that decision.

## 6. Human-in-the-loop improvement

Do not manually verify the entire corpus. After each useful model, rank samples
for review using high loss/low confidence, checkpoint disagreement, repeated
errors, transcript/audio mismatch, and underrepresented speakers/domains.

Review a meaningful batch, normally 300-1,000 samples, one completed error
category, or at least one verified hour, before retraining. Never train on gold
validation or test rows.

## 7. Compute and failure gates

Initial allowance remains five included Camber GPU hours plus at most USD 10
paid compute. Free resources are still measured.

### Gate A: local/free validation

- Dataset, metric, unit, integration, and CPU smoke checks pass.
- Prediction reports work.
- Checkpoints save, reload, and resume.

### Gate B: capped GPU smoke

- Fixed small data/step limit.
- Mixed precision, VRAM, throughput, and utilization measured.
- Checkpoint recovery verified.
- One deliberate interruption and deterministic resume verified.

### Gate C: measured pilot

- Run 500-1,000 steps or 5-10% of the data.
- Measure training, evaluation, checkpoint time, and peak memory.
- Estimate full cost with a 25% safety margin.
- Stop if projected cost exceeds the approved allowance.

### Gate D: bounded full experiment

- Advance only a pilot with meaningful measured benefit.
- Run one full winner at a time.
- Download predictions, logs, metadata, and important checkpoints before the
  environment expires.
- Analyze and document failures before retrying.

Follow the [Kaggle](../training/kaggle-cli.md),
[Colab](../training/colab-cli.md), [Camber](../training/camber-cli.md), and
[compute](compute-plan.md) procedures.

## 8. Completion criteria

The project is ready for another paid/full training run only when:

- A frozen, fingerprinted, leakage-checked dataset exists.
- Sinhala normalization is documented and tested.
- The relevant untouched/model-family baseline is recorded.
- Detailed error reports work end to end.
- The run's hypothesis and control are explicit.
- Training smoke and resume tests pass.
- A bounded pilot shows sufficient benefit.
- Pilot throughput, memory, and projected cost are measured.
- The full configuration, stop condition, and maximum spend are approved.

The project is complete when the selected system has:

- Reproducible strict and canonical test WER/CER.
- Independent, speaker-diverse evaluation.
- Subgroup and error analysis.
- English-retention and code-switch results.
- Deployment latency and resource measurements.
- Model card, reproducible configuration, artifact hashes, and actual cost.

## 9. Progress

This checklist is the durable project completion record. Do not collapse
completed items into summaries or remove them when priorities change.

### Foundation, data, and tooling

- [x] Historical run and error-profile review.
- [x] Clean-project scope and architecture recorded.
- [x] Package scaffold and development tooling.
- [x] Dataset manifest and audit tooling.
- [x] Conservative Sinhala normalization v1 and tests.
- [x] Download and fingerprint independently available upstream datasets.
- [x] Run source and cross-source audits on actual audio/transcript data.
- [x] Build the local adjudication UI and self-contained review queues.
- [x] Review and lock audio-verified validation/test sets as dataset v4.
- [x] Generate deterministic speaker-disjoint candidate splits.
- [x] Audit full-corpus boundary silence and clipping without modifying sources.
- [x] Test boundary trimming and reject it after manual review and local A/B.
- [x] Implement strict/canonical metrics, error labels, subgroups, and
  confidence intervals.
- [x] Implement local Whisper training, prediction, and reporting paths.
- [x] Enforce configuration-based cloud cost and test-set access gates.
- [x] Implement configuration-driven training.
- [x] Complete local smoke and checkpoint-resume tests.

### Data and preprocessing experiments

- [x] Complete the first controlled preprocessing experiment: trimmed versus
  original audio.
- [x] Complete controlled transcript-label refinement A/B for Sinhala-only and
  Latin-only text; reject automatic text-only refinement and leave dataset v3
  unchanged.
- [x] Audio-verify 295 disputed evaluation references and 100 unchanged
  controls; freeze 392 usable references as v4 and hold out 1,604 unheard rows.
- [x] Benchmark Bedrock-accessible transcript refiners against 293
  audio-verified targets; establish that Sonnet 4.6 is safest among those
  tested but does not reproduce the earlier ChatGPT pass.

### Baselines and completed training research

- [x] Complete capped 100-step free-Colab wide-LoRA pilot and measure
  throughput.
- [x] Evaluate untouched Whisper-small on v4 validation.
- [x] Freeze LibriSpeech test-clean for English retention and evaluate
  untouched Whisper-small.
- [x] Complete E000-E007 Whisper experiments, including the nested
  220.877-hour data-scale curve and checkpoint-resumed cloud training.
- [x] Establish teacher-behavior replay and pass English retention through
  E007.
- [x] Measure E007 at 81.71% canonical WER and 26.15% canonical CER and
  establish that scale alone on the fixed LoRA recipe will not reach target.
- [x] Complete E008 rank/learning-rate search and document its evaluation and
  concurrency failures.
- [x] Complete E010 clean validation of rank 32 and learning rate approximately
  2.345e-4.
- [x] Build E009's extended Sinhala tokenizer, measure 57.1% token compression,
  diagnose the frozen-row failure, and implement the corrected configuration.

### Current ranked work

- [x] Complete and document E007 greedy-versus-beam decoding comparison
  (greedy 81.40%/26.15%, beam5 76.26%/26.67% canonical WER/CER).
- [x] Complete per-speaker/duration/length evaluation reports for E007, the
  current best checkpoint (still owed for E004-E006, lower priority since
  they are no longer active candidates).
- [ ] Build a broader independent, speaker-diverse Sinhala evaluation set.
  Blocked on a licensing decision: the existing corpus has no unused
  speakers left (confirmed by inspection); the only candidate with real
  diversity already indexed here is the SPEAK-ASR YouTube corpus (34
  uploaders, 9.1h), whose license is unresolved.
- [ ] Run the Omnilingual zero-training model-family bake-off.
- [ ] Run a bounded Omnilingual CTC 300M adaptation pilot if justified.
- [ ] Test Sinhala LM-assisted CTC decoding if justified.
- [x] Re-run the corrected E009 tokenizer pilot -- fix confirmed correct,
  stopped per its own stop condition (still doesn't beat E010).
- [ ] Compare bounded Whisper full-parameter adaptation with rank-32 LoRA.
- [ ] Run only evidence-triggered fallback models and ablations.
- [ ] Freeze the selected candidate and run the final unopened test.
- [ ] Complete deployment benchmark and model card.
