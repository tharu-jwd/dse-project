# Evaluation Protocol

Every model is scored from a row-level prediction file containing at least
`sample_id`, `reference`, and `prediction`. Preserve manifest metadata such as
`source_dataset`, `language_class`, `speaker_id`, `duration_seconds`, and
`dataset_split` in the same file so subgroup reports remain traceable.

Strict scoring performs only Unicode NFC normalization and whitespace cleanup.
Canonical scoring applies the versioned metric normalization in
`src/sinhala_asr/text/normalizer.py`. CER excludes whitespace characters; WER
uses whitespace-delimited words. Report both metrics as ratios or percentages,
never as an unlabeled bare number.

```bash
PYTHONPATH=src python scripts/evaluation/predict.py \
  --model openai/whisper-small \
  --manifest data/versions/v4/manifest.parquet \
  --split validation \
  --output runs/untouched-small/validation-predictions.parquet

PYTHONPATH=src python scripts/evaluation/evaluate_predictions.py \
  --predictions runs/untouched-small/validation-predictions.parquet \
  --output-dir runs/RUN_ID/evaluation
```

Prediction refuses candidate splits unless a bounded `--max-rows` smoke limit
is supplied. It also refuses the locked test split unless `--unlock-test` is
explicitly supplied after model and decoding selection are frozen.

The evaluator writes scored rows, machine-readable aggregate/subgroup metrics,
95% paired bootstrap intervals, edit-operation counts, and a Markdown summary.
Its automatic error labels are triage signals, not linguistic ground truth.
Suspected reference/audio faults must be confirmed in the native-review UI.

Validation predictions may be used for recipe and checkpoint selection. Test
predictions must not be generated until the candidate, normalization version,
and decoding configuration have been frozen.

## English retention benchmark

Standalone English retention uses the complete 2,620-row LibriSpeech
[`test.clean` split](https://huggingface.co/datasets/openslr/librispeech_asr)
from `openslr/librispeech_asr` (CC BY 4.0), not English rows from the Sinhala
corpus. The immutable upstream Parquet SHA-256 is
`7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0`.
The prepared benchmark contains 5.403 hours from 40 speakers.

The preparation script verifies that source hash, expected row count, 16-kHz
mono audio, and non-empty embedded bytes. It records a semantic content
fingerprint over sample/speaker/chapter IDs, reference text, and each audio
hash. That content fingerprint—not a version-dependent Parquet serialization
hash—defines equality across local and Colab copies.

Every adapter is compared with untouched `openai/whisper-small` on these exact
rows using English transcription prompts and paired bootstrap intervals. This
benchmark is evaluation-only and is never mixed into Sinhala training unless a
later, separately documented English-replay experiment explicitly introduces a
licensed training split.

### Current English acceptance threshold

Effective 2026-09-14, the project owner accepts a candidate when its absolute
canonical English WER on this frozen benchmark is **at most 10.00%**. Strict
English WER remains diagnostic because reference casing and punctuation make it
unsuitable as the retention gate. Canonical CER, the change from untouched
Whisper-small, paired confidence intervals, and row-level regressions must still
be reported, but they no longer independently reject a candidate below the
10.00% WER ceiling.

This is a post-hoc project policy change. Earlier experiment reports preserve
their original pre-registered gate (+0.50 WER points with paired 95% upper bound
at +1.00 point); their historical decisions must not be rewritten. Current
selection uses the new absolute ceiling and labels any retrospective status
change explicitly.

| Completed checkpoint | Canonical English WER | Current status |
|---|---:|---|
| Untouched Whisper-small | 4.2338% | Pass |
| E001 | 4.2978% | Pass |
| E002 | 6.3535% | Pass (failed the original tighter gate) |
| E003 raw-reference replay | 13.84% | **Fail** |
| E004 teacher replay | 4.62% | Pass |
| E005 teacher replay | 4.52% | Pass |
| E006 teacher replay | 4.5821% | Pass |
| E007 teacher replay | 4.5971% | Pass |
| E014 5e-5/linear full tune | 5.7794% | Pass (failed the original tighter gate) |
| E014 1e-5/cosine full tune | 4.7044% | Pass |

Yohan's external 3e-5 full tune (about 80.9% English WER) remains a clear
failure under either policy. His 1e-5/cosine result is contextual evidence, not
a row-aligned project result, and is therefore not included in the table.

## Explaining metric changes

Every experiment comparison must record the run IDs, starting checkpoint,
dataset fingerprint, split, seed, preprocessing, decoding settings, and the one
intended factor changed. Report strict and canonical WER/CER for both runs,
their absolute and relative changes, paired confidence intervals, and separate
insertion, deletion, and substitution counts.

Break changes down by at least language class, duration, and transcript length;
add source, speaker, code-switch, named-entity, or other slices when relevant.
Retain sample-level improvements and regressions so the aggregate movement can
be traced to concrete utterances and error categories. If WER and CER disagree,
or one subgroup improves while another regresses, state that explicitly.

Call a result improved or regressed only when the paired evidence supports that
conclusion. A small movement inside the confidence interval is `inconclusive`,
not an improvement. Distinguish demonstrated causes from plausible mechanisms:
a controlled one-factor ablation can support attribution; an uncontrolled
comparison can only establish association. No WER/CER change may be reported
without an accompanying comparison record and explanation status.

`scripts/evaluation/compare_predictions.py` computes exactly this paired
comparison (row-matched by `sample_id`, both strict and canonical, with the
95% confidence intervals above) between two prediction files; every
experiment comparison from E002 onward uses it rather than diffing aggregate
WER/CER by hand.

## Deeper diagnostic analyses

When aggregate and subgroup metrics are not enough to explain *why* an error
rate sits where it does, a targeted local analysis on an experiment's already-
scored predictions can be more informative than another training run --
for example, tabulating character-substitution pairs to check whether error
is concentrated in a small, known confusion set versus broadly spread. These
are diagnostics, not trained experiments, and belong in `docs/audits/` with a
clear statement of what was computed, on what data, and what it does and does
not explain; see
[the E006 near-homophone error analysis](../audits/e006-near-homophone-error-analysis.md)
for the pattern.

## Interactively trying a model

`scripts/review/try_model_app.py` is a local Streamlit UI for hearing and
reading a specific completed experiment's actual behavior, not just its
aggregate numbers: pick any experiment with a locally-available final adapter
(or the untouched baseline), then feed it your own live-recorded voice
(microphone, via `st.audio_input`), uploaded audio, a real row from the
frozen 206-row Sinhala validation set, or a row from the 2,620-row
English-retention benchmark. It plays the audio, shows the raw prediction, a
word-level diff against the reference when one exists, and that one clip's
strict/canonical WER/CER. Runs fully offline once the base model is cached.
This is a qualitative complement to aggregate metrics, not a replacement --
one clip is not statistically meaningful, and it must never substitute for the
frozen, paired, confidence-interval-backed evaluation an experiment report
relies on.

The model list also includes third-party checkpoints for direct qualitative
comparison, each explicitly labeled `EXTERNAL -- <source>` so it is never
mistaken for one of this project's own experiments -- currently
Yohan2003/whisper-small-sinhala's full fine-tune. Independently re-evaluated
against this project's own frozen validation set; a full write-up covering
all four of that repo's checkpoints is pending (evaluation in progress as of
2026-09-07).

```bash
pip install -e '.[review,train]'
PYTHONPATH=src streamlit run scripts/review/try_model_app.py
```

## Colloquial/formal mismatch label

`error_labels()` can tag a substitution as `colloquial_formal_mismatch` when
the reference and prediction words reduce to the same root under the optional
SinLing stemmer (`pip install -e '.[dev]' && pip install -e '.[morphology]'` --
or add `morphology` alongside whatever other extras are already in use).
Without that extra installed, this check is skipped silently and every other
label is unaffected -- `evaluate_predictions.py` and the rest of the official
scoring path do not require it.

This fills the "Colloquial/formal mismatch" category from the plan's Phase 3
taxonomy, previously unimplemented. Like every other automatic label, it is a
triage signal, not linguistic ground truth, and the false-positive risk here
is real and observed, not theoretical: `SinhalaStemmer` is a rule-based
suffix stripper, not a dictionary-backed morphological analyzer, so it can
report a shared root for words that are not actually the same word with a
different formality register -- for example, it flagged an E006 validation
row where the prediction was simply a garbled misrecognition
(`ලැබුණු` -> `ලැබලු`), not a genuine formal/colloquial variant. Treat this
label the same way as every other one: a reason to look at the row, not a
verdict on it.
