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
report: `reports/e009-tokenizer-extension/whisper-small-si-250/extension-report.json`.

The 57.1% reduction is above the external precedent's reported 30-61% range
(Hindi 27%, Malayalam 61%), consistent with the near-homophone analysis's
prediction that Sinhala's worse starting point (literal zero vocabulary,
versus the precedent languages' 27-79 tokens/word) should mean a *larger*
relative gain, not a smaller one.

## Pilot run: destabilization confirmed (real finding, recipe needs a fix)

Ran on Camber Cloud (2026-09-06, job `25195`, `--gpu --size xsmall`, real L4)
after several environment-only failed attempts (wrong project-root path in
`download_openslr52.py`/`inventory_sources.py`/`index_openslr52.py`, a
Python-version mismatch needing a `numpy<2.2` pin, `git_commit()` crashing
outside a git checkout -- all fixed in-session, see commits `f1037d1` and
`49b8218`). The job itself completed (exit 0) but the *result* is bad, and
directly explains why:

```
step 50:  eval_wer=1.691   eval_cer=0.938   (169% / 94%)
step 100: eval_wer=9.994   eval_cer=2.423   (999% / 242%) -- got WORSE, not better
train_loss: 40.6 (started ~50, this project's normal runs start ~11-12)
```

WER *increasing* over training, plus a starting loss roughly 4-5x higher
than any other experiment in this project at the same step count, is not
"a modest improvement" -- it is instability. Downloaded `checkpoint-100`
and ran the same direct-reproduction check used for the E008 bug (load the
real checkpoint, generate on real validation clips, read the actual text,
not just the aggregate metric):

```
REF: ඒත් අපිට ලැබුණු උත්තර පට්ට පල් බොරු
HYP: ' ស ឡ វ ល ហ ផ រ ឧ ភ ឦ ឰ ប ...'   (222 tokens, ran to near the cap)
```

That is **Khmer script**, not Sinhala, not garbage byte-fallback text --
actual Khmer Unicode characters, on two of four sampled clips. The other
two produced short, wrong-but-real Sinhala fragments. This is a
structurally different failure from the E008 repetition loop (which was a
decoding-only issue on an otherwise-healthy model) -- here the model itself
is unstable.

### Root cause (structurally confirmed, not inferred)

`WhisperForConditionalGeneration` ties its input embeddings and output
projection (`model.config.tie_word_embeddings == True`;
`model.decoder.embed_tokens` and `proj_out` share one weight matrix, verified
directly). `resize_token_embeddings()` correctly grows that matrix to 52115
rows with the new 250 rows randomly initialized (the same approach the
external precedent used). But `model.decoder.embed_tokens`/`proj_out` are
**not** in this project's LoRA `target_modules`
(`q_proj,k_proj,v_proj,out_proj,fc1,fc2`) -- and `get_peft_model()` freezes
every parameter outside the target modules (and outside `modules_to_save`,
which the current recipe does not set). So the 250 new embedding rows are
frozen at their random initial values for the entire run: no gradient ever
reaches them. Every time the extended tokenizer's greedy BPE merges route
real training text through one of those 250 tokens (which is often --
that's the whole point of the extension, 57.1% fewer tokens/word), the model
is being asked to learn from/predict a token whose representation can never
move. That is a plausible, sufficient explanation for both the rising WER
and the specific failure mode (the model falling back on unrelated
high-probability tokens from its original 99-language pretraining,
including Khmer, when its Sinhala-token predictions are anchored to noise).

This is exactly the risk [the near-homophone
analysis](e006-near-homophone-error-analysis.md#external-precedent-for-this-exact-fix)
flagged before building anything ("no special handling of the
embedding/LM-head freezing concern raised above") -- confirmed now, for
this project's specific LoRA recipe, not the full-fine-tune recipe the
external paper actually used (full fine-tuning leaves nothing frozen, so
their "no instability" result was never testing this failure mode at all).

### The fix, not yet applied

PEFT's documented pattern for exactly this situation: add
`modules_to_save=["embed_tokens", "proj_out"]` to the `LoraConfig`, which
keeps those two modules fully trainable (not LoRA-adapted, plain fine-tuned)
alongside the LoRA adapters on the attention/MLP projections. Requires
plumbing a new field through `TrainConfig`/`train.py`'s LoRA branch --
currently `LoraConfig(...)` is called with no `modules_to_save` argument at
all. Not yet implemented; the pilot needs to be re-run with this fix before
concluding anything about the tokenizer extension's real effect on WER/CER.

## Status

Build done, first pilot run done and diagnosed as unstable for a specific,
confirmed, fixable reason (frozen new-token embeddings under LoRA). Recipe
fix (`modules_to_save`) not yet implemented. Second pilot run pending that
fix.
