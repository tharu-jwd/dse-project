# run1

- **Canonical name (dse-project finetune_tracker.csv):** `run1`
- **W&B run:** [run1](https://wandb.ai/yohanj-23-university-of-moratuwa/whisper/runs/peti3e5t) (id `peti3e5t`, state `finished`)
- **Error-analysis folder:** `ErrorAnalysis/run1/error_analysis/` -- originally saved under the inconsistent name `run1_full`, matched to this run by **hyperparameters** (lr, scheduler, batch size, LoRA config) and renamed to match this run's actual name.

## Hyperparameters (from W&B config)
- `learning_rate`: 3e-05
- `lr_scheduler_type`: linear
- `per_device_train_batch_size`: 8
- `gradient_accumulation_steps`: 4
- `num_train_epochs`: 4
- `warmup_steps`: 500

## Notes / results (from finetune_tracker.csv)
Full fine-tune, lr=3e-5 linear, per_device_bs=8 x grad_accum=4 (eff. 32), 4 epochs, warmup 500. Val WER 19.96%/CER 4.17%; Test WER 17.08%/CER 3.49%. English forgetting: severe (+76.59pt WER).

## Final W&B summary metrics
- `eval/wer`: 19.96063876157392
- `eval/cer`: 4.16636224154307
- `eval/loss`: 0.10654214769601822
- `train/loss`: 0.0347
- `train/epoch`: 4

## Why this run matters: it is the baseline that drove the data fix

*Two WER figures for this run, by design:* the section above quotes **17.08%**
(the tracker's `evaluate_finetuned.py` figure); this section and the
error-analysis outputs use **17.36%** (`error_analysis.py` re-scoring
`predictions.csv`). run1 is the only one of the five runs where the two scripts
disagree. The cause was not investigated. See `Analysis.md` §1a.

run1 is the **best WER of the five runs (17.36%)** and also the run whose error
analysis exposed the data defect that `stratified_v4` and `stratified_v5` were
built to fix. It was trained on `stratified` (v1), the original split, which was
not built as a speaker-disjoint split.

### What clustering its 6,128 wrong samples revealed

The "what it shows" column is my reading of each cluster's five worst examples,
not a measured label. The clusters are unsupervised (see `Analysis.md` §2c).

| Cluster | Size | Share of errors | What its worst examples show |
|---|---|---|---|
| **1** | **2,749** | **44.9%** | Spacing disagreements (largest, least distinctive cluster) |
| 3 | 857 | 14.0% | Verb-ending splits + colloquial endings |
| 7 | 633 | 10.3% | Clitic-particle gluing (`ම`, `ව`) |
| 5 | 534 | 8.7% | Compound-verb splits (`සිදු කිරීම`) |
| 2 | 401 | 6.5% | ZWJ rakaransaya conjunct (`්‍ර`) |
| 4 | 353 | 5.8% | Colloquial quotative (`කියල`/`කියලා`) |
| 6 | 323 | 5.3% | Colloquial `තියෙන`/`තියන` + spacing |
| 0 | 278 | 4.5% | ZWJ yansaya conjunct (`්‍ය`) |

**About 42% of the wrong samples were spacing disagreements, not mishearing.**
Measured directly: 2,551 of 6,128 (41.6%) become identical to the reference once
spaces and punctuation are removed, and they carry 42.6% of all word-level errors.
In these the model transcribed the right sounds and disagreed only about where
word breaks go:

| Reference | Prediction | WER |
|---|---|---|
| `තුන්වැදෑරුම්ය.` | `තුන් වැදෑරුම් ය.` | **3.00** |
| `කදාවළලු,` | `කඳා වළලු` | 2.00 |
| `බොරුකීම` | `බොරු කීම,` | 2.00 |
| `ටීවිවල` | `ටීවි වල` | 2.00 |
| `එනම්ඉතිරිය` | `එනම් ඉතිරිය` | 2.00 |

A one-word reference written as three words scores WER 3.00 — the worst severity
bucket in the dataset — for an utterance arguably transcribed correctly.

### The clearest signal: clitic particles counted as deletions

run1's top "deleted words" are not content words but single-character
grammatical particles, because the reference wrote them **separately** and the
model glued them onto the preceding word:

| Particle | Deletions | Meaning |
|---|---|---|
| `ම` | **872** | emphatic |
| `ව` | **407** | adverbial ("-ly") |
| `ද` | **289** | question / "also" |
| `දී` | 236 | locative |
| `ය` | 188 | copula |

**16 of run1's top 25 substitutions are this same glue pattern** —
`එමෙන්`→`එමෙන්ම` (39), `පිළිබඳ`→`පිළිබඳව` (36), `ඇත්තට`→`ඇත්තටම` (29),
`මෙන්`→`මෙන්ම` (28), `පසු`→`පසුව` (21), `යුතු`→`යුතුව` (20), and so on. In
context: `ref: අතිශයින් ම ප්‍රකාශිත ය` → `pred: අතිශයින්ම ප්‍රකාශිතය`, where the
prediction is arguably the *more* standard written form.

### What was done about it

Two changes, producing `stratified_v4`: **spacing/compounding normalization**
across all four source corpora, and a **speaker-disjoint re-split** (v1 was not
built as a speaker-disjoint split, so its test set may share speakers with train,
which would flatter this run's 17.36%. That effect was not measured; the parquet
files have no speaker column to measure it with). run5 re-ran this run's recipe on that data — see
[`../../run5/run_summary/README.md`](../../run5/run_summary/README.md) for the
before/after comparison, and [`../../Analysis.md`](../../Analysis.md) §3 for the
full diagnosis.

⚠️ **Do not compare this run's 17.36% directly against run5's 19.06%** — they are
measured on different test sets (15,483 samples, not speaker-disjoint vs. 15,860
samples, speaker-disjoint). See `Analysis.md` §5a.

### Caveat: catastrophic English forgetting

This run destroyed the base model's English: **80.86% English WER vs. 4.27% for
base Whisper-small (+76.59 pt, "severe")**. run4 (wider LoRA targets) and
`full-lr1e-5-e4-cosine-bs64` (lr 1e-5, cosine) both stayed within +3 pt, but those
runs differ from this one in several variables at once, so this doesn't isolate
which of them caused the loss. After the wake word, English voice commands are
classified by a separate audio classifier that works from raw audio rather than
from a Whisper transcript ([`openWake/README.md`](../../../openWake/README.md)).
The repo doesn't say whether forgetting motivated that choice.

## Contents of this folder
- `training_curves.png` - train/eval loss, eval WER, eval CER vs. step (from W&B history)
- `lr_schedule.png` - learning-rate schedule vs. step
- `wandb_config.json` / `wandb_summary.json` - full W&B run config/summary
- `predictions.csv` - reference/prediction pairs used for error analysis (copied from `ErrorAnalysis/eval_results/`)
- `error_analysis/` - clusters.txt, confusions.txt, errors_by_severity.csv (copied from the existing local error-analysis output)