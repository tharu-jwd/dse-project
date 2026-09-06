# E009: Sinhala tokenizer vocabulary extension, bounded pilot

## Status

Pilot run complete; **result is instability, not a valid Sinhala WER
measurement**. Root cause confirmed and fixed in code
(`modules_to_save`/`ensure_weight_tying` added to the LoRA config). Re-run
with the fix not yet done.

## Question

Does extending `openai/whisper-small`'s tokenizer with 250 dedicated
Sinhala subword tokens (versus its stock byte-level fallback, which has
zero dedicated Sinhala vocabulary) reduce WER/CER at this project's
established 100-step bounded-pilot scale, without destabilizing the base
model -- the check called for before this lever could be scoped for a real
experiment?

## Motivation

Two independent findings converged on this fix for this project's single
worst confusion pair (ל/ළ da/dda): [the near-homophone error
analysis](../audits/e006-near-homophone-error-analysis.md) found (a) the
pair is one phoneme under Google's G2P mapping, and (b) `whisper-small`'s
tokenizer has zero dedicated Sinhala vocabulary at all -- every Sinhala
codepoint is two raw UTF-8 byte-level fallback tokens. Real external
precedent (ICASSP 2025, 8 Indic-script languages) found 250 new tokens
optimal and reported no training instability -- from full fine-tuning,
where nothing is frozen, not this project's LoRA recipe.

## What was built

`scripts/training/extend_tokenizer.py`: trains a throwaway tokenizer via
`train_new_from_iterator()` over the real v4 train-split corpus (182,665
rows), diffs its vocabulary against the base tokenizer, appends the top
250 genuinely-new tokens via `add_tokens()` -- purely additive. A real bug
was found and fixed before trusting the result: `add_tokens()` silently
no-ops on the tokenizer's internal byte-remapped strings; needed the text
decoded back via `convert_tokens_to_string()` first. Measured on the real
corpus: **57.1% fewer tokens/word** (10.214 -> 4.386), above the external
precedent's reported 30-61% range. Wired into `train.py`/`TrainConfig` via
`extended_tokenizer_path`.

## Frozen recipe (pilot)

Identical to E001: rank=16, lr=5e-5, 100 steps, full v4 manifest, same
206-row frozen validation set -- only `extended_tokenizer_path` added, so
the result is directly comparable to E001's own real number at this step
budget.

## Result: unstable, root cause confirmed

```
step 50:  eval_wer=169%   eval_cer=94%
step 100: eval_wer=999%   eval_cer=242%   -- got WORSE, not better
train_loss: 40.6 (this project's normal runs at this step count start ~11-12)
```

Downloaded `checkpoint-100` and ran the same direct-reproduction check used
for the E008 bug: on 2 of 4 sampled validation clips, the model generated
actual **Khmer script**, not Sinhala at all.

Root cause confirmed structurally, not guessed: `WhisperForConditionalGeneration`
ties its input embeddings and output projection
(`model.config.tie_word_embeddings == True`, verified directly). Neither
is in this project's LoRA `target_modules`, so `get_peft_model()` freezes
them by default -- meaning the 250 newly-added embedding rows stayed at
their random initialization for the entire run, since `modules_to_save`
was not set. Every time the extended tokenizer routed real training text
through one of those 250 tokens (which is often, by design), the model
was being asked to learn from a representation that could never move.
Full trace: [the E009 audit](../audits/e009-tokenizer-extension.md).

## Fix (committed, not yet re-run on real data)

Adds `modules_to_save=["embed_tokens", "proj_out"]` to the LoRA config,
active only when `extended_tokenizer_path` is set (every other
experiment's LoRA config is byte-for-byte unchanged -- verified directly:
identical trainable parameter count). Also sets `ensure_weight_tying=True`
so PEFT keeps one shared trainable copy of the tied matrix instead of two
independent ones (verified via `named_parameters()` inspection).

## Conclusion and next step

The tokenizer-extension technique itself is not disproven -- it failed for
a specific, now-fixed, structural reason specific to this project's LoRA
recipe, not because the idea is wrong. Re-run the E009 pilot with the fix
before drawing any conclusion about the technique's real effect on WER/CER.
