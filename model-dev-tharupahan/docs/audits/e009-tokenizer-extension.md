# E009: Sinhala tokenizer extension (build phase; pilot not yet run)

Evidence grade: **Verified artifact** for the tokenizer-extension numbers
below (measured directly against the real v4 corpus, bug found and fixed
in-session); the pilot training run itself is **not yet executed** -- see
Status.

## Why

Item 3 in [the plan's priority order](../project/plan.md). Two independent
findings converged on this fix for this project's single worst confusion
pair (ල/ළ): the [near-homophone error
analysis](e006-near-homophone-error-analysis.md) found (a) ල and ළ are the
same phoneme per Google's G2P mapping, and (b) `openai/whisper-small`'s
tokenizer has **zero** dedicated Sinhala vocabulary -- every Sinhala
Unicode codepoint is two raw UTF-8 byte-level fallback tokens, and ල/ළ is
the one confusion pair in that analysis whose token IDs share no structure
at all. Real external precedent
([ICASSP 2025, 8 Indic-script languages](https://arxiv.org/abs/2412.19785))
found 250 new BPE tokens learned from language-specific data to be the
optimal point on a 125/250/500/1000 sweep, random initialization for the
new embedding rows, and no reported training instability.

## What was built

`scripts/training/extend_tokenizer.py`: trains a throwaway tokenizer via
the base tokenizer's own `train_new_from_iterator()` over this project's
real train-split transcripts (182,665 rows, `data/versions/v4/manifest.parquet`),
diffs its vocabulary against the base tokenizer's, and appends the top 250
genuinely-new tokens to the *original* tokenizer via `add_tokens()` --
purely additive, no existing token/ID/merge is touched. Wired into
`scripts/training/train.py` and `TrainConfig` via a new optional
`extended_tokenizer_path` field: when set, the extended tokenizer replaces
`processor.tokenizer` and `model.resize_token_embeddings()` is called (new
rows get the model's default random init, matching the external precedent).
Both existing and new experiments that leave the field unset are completely
unaffected -- verified by the full test suite passing unchanged (63 tests).

### A real bug found and fixed before trusting the result

First run reported "250 tokens added" successfully but a **0.0% change** in
tokens-per-word on real text -- caught because the script measures that
number directly rather than trusting `add_tokens()`'s return value alone.
Root cause: `train_new_from_iterator()`'s vocabulary is expressed in the
tokenizer's internal byte-level alphabet (Sinhala `ය` comes back as the
3-character string `"à¶º"`, not `ය` itself) -- that's what the BPE merge
algorithm operates on internally, but `add_tokens()` matches literal,
human-readable text against the *raw* input, before that remapping ever
runs. Adding the byte-remapped form is a silent no-op -- it can never match
anything in real text, no error raised. Fixed by decoding each candidate
through `convert_tokens_to_string()` before adding it, which reverses the
byte-level remapping back to the real character(s). Confirmed the fix
directly: `convert_tokens_to_string(["à·Ĭ"])` returns `"්"` (a real Sinhala
vowel sign), not the byte-remapped string.

## Result (measured, not the pilot -- this is the tokenizer only)

```
base tokenizer vocab size: 51865
extended vocab size:       52115  (+250)
tokens/word before: 10.214
tokens/word after:   4.386
reduction: 57.1%
```

The first 18 selected tokens are single Sinhala consonants/vowel-signs
(``ි`` ``ය`` ``ර`` ``ක`` ... -- each one alone was previously *two* byte
tokens, so a single new token per character is already a 2x win on its
own), followed by space-prefixed word-initial variants, consistent with
what a frequency-driven BPE trainer should surface first. Full list and
report: `artifacts/tokenizer-extension/whisper-small-si-250/extension-report.json`.

The 57.1% reduction is above the external precedent's reported 30-61% range
(Hindi 27%, Malayalam 61%), consistent with the near-homophone analysis's
prediction that Sinhala's worse starting point (literal zero vocabulary,
versus the precedent languages' 27-79 tokens/word) should mean a *larger*
relative gain, not a smaller one.

## Status

Build and unit-level verification done; **the actual bounded pilot training
run has not been executed yet.** Config is ready:
`configs/training/experiments/e009-tokenizer-extension-pilot-v4.json` --
identical recipe to E001 (rank=16, lr=5e-5, 100 steps) plus
`extended_tokenizer_path`, so its result is directly comparable to E001's
already-known number (114.26% strict WER at this same tiny step budget) as
the destabilization check the plan called for. Deliberately held rather
than run locally: the local machine was in active interactive use, and a
100-step LoRA run at local MPS speed would visibly lag it (confirmed
firsthand earlier this session with the E008 search). Will run on Camber
Cloud once its current data upload for E008 finishes.
