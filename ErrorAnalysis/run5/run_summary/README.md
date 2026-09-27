# run5

- **Canonical name (dse-project finetune_tracker.csv):** `run5`
- **W&B run:** [run7-v4-lr3e-5-bs64-v3-resume](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/qww5akos) (id `qww5akos`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run5/error_analysis/` -- originally saved under the inconsistent name `run5_full`, matched to this run by **hyperparameters** (lr, scheduler, batch size, LoRA config) and renamed to match this run's actual name.

## Resume chain (predecessor W&B attempts)
This run is the last link in a 4-attempt chain, all logged as separate W&B runs under the (misleading) name `run7-v4-lr3e-5-bs64*`. Only the final attempt finished and is the one used for evaluation/error analysis above.
1. `8bpxy2xm` -- run7-v4-lr3e-5-bs64 -- **failed** -- 2026-09-08T11:36:15Z (bs=64, grad_accum=1)
2. `rot9ket4` -- run7-v4-lr3e-5-bs64 -- **failed** -- 2026-09-08T11:37:59Z (bs=16, grad_accum=4)
3. `f9no0ylf` -- run7-v4-lr3e-5-bs64-v2 -- **crashed** -- 2026-09-08T13:33:42Z (bs=8, grad_accum=8)
4. `qww5akos` -- run7-v4-lr3e-5-bs64-v3-resume -- **finished** -- 2026-09-09T13:11:51Z (bs=8, grad_accum=8) — resumed from checkpoint-5778 of attempt 3 after a transformers/torch.load (CVE-2025-32434) version incompatibility; this is the run reported on in this folder.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 8
- `gradient_accumulation_steps`: 8
- `num_train_epochs`: 4
- `warmup_steps`: 500

## Notes / results (from finetune_tracker.csv)
Full fine-tune, lr=3e-5 linear, per_device_bs=8 x grad_accum=8 (eff. 64), 4 epochs, warmup 500. Resumed from checkpoint-5778 after a transformers/torch.load (CVE-2025-32434) incompatibility forced a pod switch and a pinned-version fix. load_best_model_at_end=True (metric=wer). Val WER 22.41%/CER 6.65%; Test WER 19.06%/CER 4.90% (test notably better than val, unlike other runs). English forgetting check skipped for this run (not evaluated).

## Final W&B summary metrics
- `eval/wer`: 22.41171954964176
- `eval/cer`: 6.646411785307181
- `eval/loss`: 0.14322230219841003
- `train/loss`: 0.0598
- `train/epoch`: 3.9999350691513538

## Why this run matters: it tests whether the data fix worked

run5 re-ran **run1's recipe** (full fine-tune, lr 3e-5 linear, 4 epochs, warmup
500) on **`stratified_v4`** instead of `stratified` (v1). v4 differs from v1 in
two deliberate ways: **spacing/compounding normalized** across all four source
corpora, and **speaker-disjoint** train/validation/test splits.

### ⚠️ Read this before comparing 19.06% against run1's 17.36%

**These two numbers are measured on different test sets and are not comparable.**

| | run1 | run5 |
|---|---|---|
| Test samples | 15,483 | **15,860** |
| Test split | `stratified` (v1) | `stratified_v4` |
| Speaker-disjoint? | **No** | **Yes** |
| WER | 17.36% | 19.06% |

run1's test set contained speakers the model had trained on, so part of its
17.36% was speaker memorization. run5's 19.06% is measured entirely on **voices
the model had never heard**. A higher number on a harder, honest test is not a
worse model.

Two further confounds to disclose: run5 used effective batch 64 (run1 used 32),
and it was resumed mid-training from `checkpoint-5778` after a version
incompatibility forced a pod switch (see "Resume chain" above).

### The data fix demonstrably worked

The error-type split shows the trade directly:

| | Substitutions | Deletions | Insertions |
|---|---|---|---|
| run1 | 11.47% | **4.68%** | 1.21% |
| run5 | 14.67% | **2.33%** | 2.07% |

**Deletions halved** — that is the spacing fix landing. **Substitutions rose** —
that is the cost of unseen speakers (genuine acoustic confusions on new voices).

run1's dominant error class was clitic particles written separately in the
reference but glued on by the model. Those collapsed:

| "Deleted" particle | run1 | run5 | Change |
|---|---|---|---|
| `ම` | 872 | **80** | **−91%** |
| `ව` | 407 | **48** | **−88%** |
| `ද` | 289 | **73** | −75% |
| `දී` | 236 | **11** | **−95%** |
| `ය` | 188 | **106** | −44% |
| **Top-5 total** | **1,992** | **318** | **−84%** |

Same story in the substitution table — the glue pattern (`එමෙන්`→`එමෙන්ම`,
`පිළිබඳ`→`පිළිබඳව`, …) was **16 of run1's top 25** substitutions and is **1 of
run5's top 25** (only `යුතු`→`යුතුය`). The error class that was 45% of run1's
failures is effectively gone.

### What its re-clustering shows is left

| Cluster | Size | What it is |
|---|---|---|
| 0 | 2,656 | Mixed: register + rare words + proper nouns |
| 7 | 1,255 | Mixed residual |
| 2 | 1,038 | Verb endings, register |
| 6 | 993 | Mixed residual |
| **5** | **652** | **ZWJ rakaransaya (`්‍ර`) — survived the v4 pass** |
| 1 | 472 | **Register (`කියල`/`කියලා`)** |
| 3 | 280 | Compound-verb splits |
| **4** | **130** | **Honorific/religious register (`වහන්සේ`)** |

**The new #1 residual is register inconsistency — and it is provably a label
defect, not an acoustic one,** because the same pair errs in *both directions*:

| Pair | A → B | B → A |
|---|---|---|
| `කියල` / `කියලා` | 91 | 23 |
| `එය` / `ඒ` | 20 | 18 |
| `සමග` / `සමඟ` | 16 | 13 |

A model that mishears a word does not mishear it symmetrically. Corpus counts
confirm the cause: `සමග` appears 353 times and `සමඟ` 345 — a coin flip, so the
model gets no signal about which form to emit.

ZWJ conjuncts also survived v4 (cluster 5): `ref: ක්‍රියාකළ යුතුයි.` →
`pred: ක්‍රියා කළ හේතුවෙයි.` — the compound split at the conjunct boundary.

### What was built in response

[`../NORMALIZATION_APPROACH.md`](../NORMALIZATION_APPROACH.md) documents the
follow-up work this diagnosis drove: **407 register-spelling pairs** and **60
word-boundary/ZWJ pairs**, applied to 13,120 of 154,828 rows and published as
**`stratified_v5`**. Its effect on WER is **unmeasured** — the next fine-tune has
not been run. Supporting artifacts in this folder's parent:
`wrong_to_correct_clean.csv`, `word_boundary_wrong_to_correct.csv`,
`normalization_diff.csv`, `word_boundary_zwj_occurrences.csv`.

Also verified here: the ~10% of sentences whose *text* appears in more than one
split were audio-hashed, and **zero** share the same audio file across splits —
so v4's speaker-disjointness holds and this run's WER is not inflated by
leakage (`NORMALIZATION_APPROACH.md` §7).

## Contents of this folder
- `training_curves.png` - train/eval loss, eval WER, eval CER vs. step (from W&B history)
- `lr_schedule.png` - learning-rate schedule vs. step
- `wandb_config.json` / `wandb_summary.json` - full W&B run config/summary
- `predictions.csv` - reference/prediction pairs used for error analysis (copied from `ErrorAnalysis/eval_results/`)
- `error_analysis/` - clusters.txt, confusions.txt, errors_by_severity.csv (copied from the existing local error-analysis output)