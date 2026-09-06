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

## External grounding: these pairs are not just visually similar

Checked against an independent reference -- Google's public Sinhala
pronunciation-rules file
([`si-si_FONIPA.txt`](https://github.com/google/language-resources/blob/master/si/si-si_FONIPA.txt),
fetched and its raw rule lines verified directly, not taken from a summary):

```
ල → l;      ළ → l;      (both map to the same phoneme, /l/)
ණ → n;      න → n;      (both map to the same phoneme, /n/)
ත → t;      ථ → t;      (both map to the same phoneme, /t/)
ද → d;      ධ → d;      (both map to the same phoneme, /d/)
ෙ → e;      ේ → eː;     (distinct: short vs long e)
ි → i; ී → iː; (distinct: short vs long i)
```

Two of the pairs this analysis found most confused --  ල/ළ and ණ/න -- are
mapped to the **identical phoneme** by this reference, not merely similar
ones. The file's own header describes it as a simplified phonetic
transcription that intentionally merges some script-level distinctions, so
this is not proof the two are truly indistinguishable in all careful
phonetic analyses of Sinhala -- but it is independent evidence, from a
source with no connection to this project, that they are close enough to
collapse in a working phonetic model built for speech applications.

This refines, not just confirms, the original hypothesis. The two vowel-length
pairs (ෙ/ේ, ි/ී) remain genuine acoustic-duration contrasts -- exactly the
kind of thing pace, training exposure, or acoustic-encoder capacity could
plausibly affect. The two consonant pairs (ල/ළ, ණ/න) are a different, likely
harder case: if a phoneme-level reference treats them as the same sound, no
amount of acoustic training can teach a model to choose between them from
audio alone in cases where they are genuinely homophonous -- the correct
choice there depends on knowing *which word it is* (lexical/orthographic
knowledge), not on hearing something more clearly. That points toward the
decoder's language-modeling behavior and tokenization, not the audio encoder,
as the more relevant lever for this specific pair of confusions -- worth
keeping in mind for the tokenizer check already queued next.

## Tokenizer check: Whisper-small has zero dedicated Sinhala vocabulary

Checked directly against the real `WhisperTokenizer` for `openai/whisper-small`
(the exact tokenizer every experiment in this project uses), not inferred.

Every one of the 128 Sinhala Unicode codepoints (U+0D80-U+0DFF) was encoded
individually. All 128 require exactly 2 tokens each -- there is no single
codepoint with its own dedicated token. Scanning the full 51,865-token
vocabulary for any token that decodes, on its own, to clean Sinhala text
(rather than a stray UTF-8 byte fragment, which decodes as `�`) found
**zero**. Every Sinhala character is represented purely as two raw UTF-8
byte-level fallback tokens -- the standard GPT-2-style byte-BPE tokenizer has
learned no Sinhala-specific subword structure at all. This means the decoder
has to learn Sinhala orthography almost entirely from these low-level byte
patterns, on top of the acoustic-to-text mapping, with none of the head start
a dedicated subword vocabulary would give a higher-resource language.

The five confusion pairs from this analysis, by their exact token IDs:

| Pair | Token IDs | Share a token? |
|---|---|---|
| ණ / න | `[16204,104]` / `[16204,109]` | yes -- same first token |
| ෙ / ේ | `[17811,247]` / `[17811,248]` | yes -- same first token, adjacent second |
| ි / ී | `[17811,240]` / `[17811,241]` | yes -- same first token, adjacent second |
| ත / ද | `[16204,255]` / `[16204,107]` | yes -- same first token |
| **ල / ළ** | `[16204,121]` / `[17811,227]` | **no -- completely disjoint** |

Four of the five pairs at least share their first byte-token (an accident of
adjacent Unicode codepoints falling in the same byte-BPE fallback bucket, not
a learned relationship) -- some structural proximity for the model to lean on.
**ල/ළ, the single most frequent confusion in this analysis (35 occurrences,
more than any other pair), is the one exception**: fully disjoint token IDs,
sharing nothing. Combined with the earlier phoneme-identity finding, this pair
gets no help from either the acoustic signal (same phoneme) or the token
representation (no shared structure) -- the two independent weakest links
this analysis found line up on the exact same pair.

This is a real, structural lever: extending the tokenizer with dedicated
Sinhala subword tokens (and resizing the model's token embeddings
accordingly) before fine-tuning, rather than relying on the base byte-level
fallback throughout. It needs its own bounded pilot to check it does not
destabilize what the base model already knows. See
[the plan's next-step priority order](../project/plan.md#next-step-priority-order-after-e006e007)
(item 3) for where this sits relative to the rank/LR search and the
full-parameter pilot, and why.

## How common are the pairs this analysis is about, versus the missing coverage?

Cross-checked against a syllable-frequency table over a large natural Sinhala
text corpus (~5.79 million syllable occurrences, 4,598 distinct syllables;
used here only as reference statistics for this cross-check, not vendored
into this repository or used as training data). Two things worth separating:

The confusion pairs this analysis is actually about are common, high-impact
characters, not edge cases:

| Character | Share of corpus |
|---|---:|
| ි (short i) | 14.29% |
| න (dental n) | 12.60% |
| ත (t) | 8.42% |
| ෙ (short e) | 6.82% |
| ද (d) | 6.29% |
| ේ (long e) | 4.10% |
| ල (dental l) | 2.46% |
| ණ (retroflex n) | 2.20% |
| ළ (retroflex l) | 1.00% |
| ී (long i) | 1.42% |

The 9 characters the earlier findings section noted as *missing* from this
validation set's coverage are, by contrast, genuinely rare -- confirming that
gap was correctly deprioritized rather than just asserted:

| Character | Share of corpus | Frequency rank (of 4,598 syllables containing it) |
|---|---:|---:|
| ධ | 1.12% | best rank 63 |
| ඥ | 0.29% | best rank 100 |
| ෘ | 0.28% | best rank 222 |
| ඝ | 0.15% | best rank 171 |
| ඬ | 0.07% | best rank 286 |
| ෛ | 0.04% | best rank 558 |
| ඕ | 0.03% | best rank 308 |
| ඊ | 0.01% | best rank 649 |
| ඓ | 0.0009% | best rank 1,626 |

This is quantified confirmation, not new discovery: it turns the earlier
"these are rare Sanskrit/Pali-loanword letters" claim into a measured one, and
shows the actual confusion pairs driving E006's error are the opposite of an
edge case -- they are among the most common characters in the language.

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
