# E009: Sinhala tokenizer vocabulary extension, bounded pilot

## Status

**Stopped, per this plan's own stop condition** (item 6:
"stop this path unless it materially beats E010 without output
instability"). The structural fix (`modules_to_save`/`ensure_weight_tying`)
is confirmed real and correct -- the wrong-script (Khmer) failure is gone
-- but the corrected rerun still does not produce usable Sinhala output at
this pilot's 100-step budget, and still does not beat E010. See
"Corrected re-run" below.

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

## Corrected re-run

Camber job `25203`, identical recipe and config
(`configs/training/experiments/e009-tokenizer-extension-pilot-v4.json`,
same seed 20260903, same 206-row validation set), same environment
pattern already verified working by job `25195`. Completed in 11.25
minutes (created_at-to-finished_at).

```
step 50:  eval_wer=100.9%  eval_cer=91.01%
step 100: eval_wer=100.1%  eval_cer=85.63%
train_loss: 22.16 (job 25195's original unfixed run: 40.6 and still
  rising -- this run's loss is materially lower and stable, a real
  difference, not noise)
```

`all_results.json`/`run-metadata.json` downloaded and hash-verified;
`final/adapter_model.safetensors` sha256
`dccba4cb9ba8a85e24695b8f5b6f52f1f7642af8d408db854bee20a46ea3c8cd`
(940.9 MB -- much larger than a normal LoRA adapter's ~26 MB, expected:
`modules_to_save` now saves the full dense `embed_tokens`/`proj_out`
matrices, not low-rank deltas).

A direct local spot-check (same method used to catch the original
Khmer-script failure) loaded this adapter and generated real predictions
for 5 validation clips:

```
REF: ඒත් අපිට ලැබුණු උත්තර පට්ට පල් බොරු
HYP: ම්්්ි්්න්්ා්්
REF: එහෙත් එය නිවැරදිව
HYP: ම්්්ි්්න්්
REF: හසන්ත සහෝගේ උපසිරැසි
HYP: ම්්්ි්්න්්
REF: ඒක ඔප්පු කරන්න ආධාර ඉල්ලුවෙ
HYP: ම්්්ි්්න්්
REF: එයාලගේ සාරය අරගන්නවා
HYP: ම්්්ි්්න්්ා්්
```

The wrong-*script* failure is gone -- output is genuinely Sinhala
characters now, not Khmer -- confirming the structural fix did what it
was supposed to. But the model has collapsed to a short, near-identical
repeated loop of bare Sinhala vowel-sign/virama marks across every input,
regardless of the reference -- not real transcription, a different kind
of degenerate output. `eval_wer`/`eval_cer` (~100%/86-91%) reflect this
accurately; they are not measurement artifacts this time, matching what
the real per-row text shows.

## Conclusion and next step

The frozen-embedding root cause is confirmed fixed -- the fix itself was
correct and is worth keeping in `train.py` regardless of this technique's
fate, since it makes any future vocabulary-extension experiment possible
in principle. But the corrected pilot still does not produce usable
output at this exact 100-step/lr=5e-5 budget, and does not beat E010's
real rank-32 result on any metric. Per this plan's own stop condition for
this item ("stop this path unless it materially beats E010 without output
instability"), **this path is stopped, not escalated to a longer/costlier
run** -- the pilot budget is the same one E001/E010 use precisely so a
result is comparable and decision-worthy without spending more than a
bounded pilot's cost, and this result is decisive: no benefit, and a
still-present (if different) instability. The most likely reason 100
steps isn't enough specifically for this technique: newly added embedding
rows start from a smart (mean/covariance) initialization but the 250 new
tokens are used often, by design, across nearly every training sequence --
adapting a meaningful fraction of a tied embedding/output matrix within
~1,600 effective training examples (100 steps x batch 4 x grad_accum 4)
looks like too little signal, unlike this project's ordinary LoRA deltas
on already-pretrained weights. A future attempt would need a materially
larger step budget to even test the technique fairly, which only makes
sense if a materially higher-priority lever runs out first -- see
[the plan's ranked order](../project/plan.md).
