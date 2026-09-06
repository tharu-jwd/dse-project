# E006 Near-Homophone Error Analysis

Diagnostic analysis, not a trained experiment. Computed entirely locally from
E006's already-downloaded, hash-verified Sinhala validation predictions
(`reports/experiments/e006-scale-100h-teacher-replay/attempts/kaggle-training-001/output/e006-training/sinhala-validation-predictions.parquet`,
206 rows). No GPU time or new training was used.

## Question

E006 (100.0 hours) reached 84.48% canonical WER / 28.74% canonical CER. The
large WER-CER gap suggested many errors might be small, spelling-level
mistakes rather than broad recognition failure. This analysis asks: how much
of E006's WER/CER is attributable to a small set of known Sinhala
near-homophone script confusions, versus genuinely broader error?

## Method

For each of the 206 validation rows, reference and prediction were normalized
with the project's own `metric_normalize` (matching official canonical
scoring), then character-aligned with `difflib.SequenceMatcher` to extract
per-row edit distance and substitution pairs. Word- and character-level
WER/CER were recomputed with the project's own `edit_counts` function (same
function official scoring uses) to keep numbers directly comparable to the
reported E006 result.

## Findings

- **5 of 206 rows are exact matches; 48 more are "near-miss" (≤3 total
  character edits)** — about a quarter of the validation set is very close to
  correct, not broadly wrong.
- The most frequent individual character-substitution pairs are dominated by
  a small set of well-known Sinhala near-homophone confusions:

  | Pair | Count | What it is |
  |---|---:|---|
  | ල ↔ ළ | 35 | dental *l* vs retroflex *ḷ* |
  | ී ↔ ි | 21 | long vs short *i* vowel sign |
  | ෙ ↔ ේ | 15 | short vs long *e* vowel sign |
  | ණ ↔ න | 17 | retroflex *ṇ* vs dental *n* |
  | ත ↔ ද | 10 | unvoiced vs voiced dental stop |

## Counterfactual: how much WER/CER would forgiving these buy?

Two counterfactuals, both computed on the same 206 rows with the same
official scoring function, folding the named character(s) to a single
representative in both reference and prediction before scoring:

| Fold applied | Canonical WER | Canonical CER |
|---|---:|---:|
| None (actual E006 result) | 84.48% | 28.74% |
| Only the 5 named linguistically-coherent pairs above | 82.01% (-2.47pp) | 26.19% (-2.55pp) |
| **Every character-substitution pair occurring ≥4 times (34 pairs, the honest ceiling of this approach)** | **78.31% (-6.17pp)** | **18.85% (-9.89pp)** |

## Conclusion

Near-homophone confusion is real, and it is the single most common
*individual* character-level error type — but it is **not** the explanation
for E006's WER. Even the maximally generous version (folding every frequent
character-substitution pair, not just the 5 clearly-linguistic ones) only
recovers about 6 WER points and 10 CER points. The remaining ~78% WER / ~19%
CER is genuinely broader recognition error (larger word-level mistakes,
insertions, deletions), not a scoring artifact.

## Priority implication

This is real, useful evidence for a *later* optimization pass (tokenizer
handling of these characters, or a correction batch targeted at rows
containing these specific confusions rather than a random sample), but it is
low-leverage relative to the other levers in
[the plan's next-step priority order](../project/plan.md#next-step-priority-order-after-e006e007).
Note the historical full-parameter checkpoint's reported ~17% WER is no longer
usable even as a rough target -- [the historical audit](historical-audit.md)
has since confirmed its evaluation split had pervasive speaker leakage.
Address this near-homophone fix only after a higher-leverage lever (a properly
scoped full-parameter pilot, or the cheaper rank/LR ablation) has been tried,
so the same fix is applied against a much lower baseline where it has
proportionally more value.

## Addendum: informal live-speech pace observation (unverified, anecdotal)

While interactively testing the models via `scripts/review/try_model_app.py`'s
live-microphone tab, the project owner observed qualitatively different
transcription behavior when reading the same content at a noticeably faster
versus slower pace than normal (clips under 5 seconds each, well within the
30-second input window and 64-token output cap used for official validation,
so this is not the input/output truncation artifact that a longer multi-
sentence recording would risk). No systematic recording, transcript logging,
or quantified comparison was made -- this is a single informal observation,
not a controlled test, and must not be treated as a confirmed model property.

Hypothesized mechanism, grounded in but not independently verified beyond the
finding above: the two most common confusion classes identified in this
analysis -- vowel-length pairs (ෙ/ේ, ි/ී) and dental/retroflex consonant pairs
(ල/ළ, ණ/න) -- are distinguished primarily by duration and formant cues that
speaking pace directly affects. Faster speech compresses vowel duration,
exactly the cue separating short and long vowel signs; unusually slow speech
exaggerates duration the other way. Both move the acoustic signal away from
whatever pace distribution the LoRA adapter's training data (OpenSLR52's
natural reading pace) actually covers. Whisper's lack of an explicit
voice-activity detector is also a documented source of hallucination/
repetition around atypically long inter-word pauses in very slow speech.

**Status: hypothesis, not evidence.** A cheap, no-GPU follow-on test would
read the same fixed sentence(s) at a controlled set of paces (for example,
three repeats each at self-rated slow/normal/fast), transcribe and score them
the same way as any other prediction, and compare the resulting WER/CER and
error-label breakdown like any other paired comparison. Not yet scheduled;
lower priority than the items in the plan's next-step order.
