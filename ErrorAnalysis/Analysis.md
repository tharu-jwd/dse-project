# Error Analysis — Whisper Small Sinhala fine-tuning (5 runs)

This is the analysis document for the Sinhala ASR fine-tuning work: what each run
scored, how the failures were grouped and diagnosed, which data defects that
diagnosis exposed, what was changed in response, and whether the change actually
worked.

It is organized as the actual chain of reasoning that was followed, because the
interesting part of this work is not any single WER number — it is that the
headline WER **went up** (17.36% → 19.06%) while the specific error class being
targeted **collapsed by 84%**, and understanding why both of those are true at
once is the whole point.

## The arc in one page

| Step | What was done | Outcome |
|---|---|---|
| 1 | Fine-tuned 5 configurations (full FT vs. LoRA variants) on `stratified` (v1) | Best: **run1, 17.36% WER** (full fine-tune) |
| 2 | Clustered run1's 6,128 wrong samples to find *what kind* of errors they were | Measured directly: **2,551 of 6,128 (41.6%)** differ from the reference *only* by spaces/punctuation, and carry **42.6%** of all word-level errors: spacing disagreement, not mishearing |
| 3 | Fixed the data: spacing normalization + **speaker-disjoint** re-split → `stratified_v4` | — |
| 4 | Re-ran the same recipe as run1 on the new data → **run5, 19.06% WER** | Headline WER rose, but the targeted class collapsed (see §5) |
| 5 | Re-clustered run5 | Spacing-only wrong samples fell from 41.6% to **13.6%**, and the particle-deletion counts fell 84%; the new #1 residual is **register/spelling inconsistency**, and it shows up **bidirectionally**, which points to a label defect rather than an acoustic one |
| 6 | Built and applied register + boundary normalization → `stratified_v5` (published) | Next fine-tune pending; see [`run5/NORMALIZATION_APPROACH.md`](run5/NORMALIZATION_APPROACH.md) |

The crucial methodological point, stated up front so no chart in this document
is read the wrong way: **run1's 17.36% and run5's 19.06% are not measured on the
same test set.** run1 was evaluated on 15,483 samples from `stratified` (v1),
run5 on 15,860 samples from `stratified_v4`, which is speaker-disjoint. Any
direct "17 → 19, so it got worse" reading is wrong. §5 unpacks this.

## Folder layout

Each run has its own top-level folder, named `run1`–`run5` in the order the runs
were executed, containing:

- `error_analysis/` — `clusters.txt` (themed failure groups), `errors_by_severity.csv`
  (every wrong sample, worst first), `confusions.txt` (top substitution / deletion /
  insertion words, plus the severity histogram)
- `run_summary/` — `predictions.csv` (raw reference/prediction pairs), `README.md`,
  `training_curves.png`, `lr_schedule.png`, `wandb_config.json`, `wandb_summary.json`

`comparison_summary.csv` (one row per run) and this file sit at the top level.

| Folder | `finetune_tracker.csv` run name | Date | Trained on |
|---|---|---|---|
| `run1` | run1-lr3e-5-bs32 | 2026-08-18 | `stratified` (v1) |
| `run2` | run6-lr3e-5-bs32_amd | 2026-08-20 | `stratified` (v1) |
| `run3` | run2-lr3e-5-bs32_lora | 2026-08-25 | `stratified` (v1) |
| `run4` | run3-lr1e-4-r32-lora | 2026-08-26 | `stratified` (v1) |
| `run5` | run5-v4-lr3e-5-bs64-v3-resume | 2026-09-09 | **`stratified_v4`** |

---

# 1. The runs and how they compare

## 1a. Headline numbers

Corpus-level WER and the error-type breakdown, from `comparison_summary.csv`:

| Run | Type | Test samples | Wrong samples | **WER** | Substitutions | Deletions | Insertions |
|---|---|---|---|---|---|---|---|
| **run1** | Full fine-tune | 15,483 | 6,128 (39.6%) | **17.36%** | 11.47% | 4.68% | 1.21% |
| **run5** | Full fine-tune (on v4) | 15,860 | 7,476 (47.1%) | **19.06%** | 14.67% | **2.33%** | 2.07% |
| **run2** | LoRA (AMD hw) | 15,483 | 7,522 (48.6%) | 21.01% | 15.80% | 2.89% | 2.32% |
| **run4** | LoRA (r=32, wide targets) | 15,483 | 8,910 (57.5%) | 25.99% | 19.67% | 3.49% | 2.83% |
| **run3** | LoRA (best-epoch2) | 15,483 | 13,993 (90.4%) | 59.07% | 45.22% | 11.22% | 2.63% |

**One number to state carefully: run1's WER is 17.36% here but 17.08% in
`finetune_tracker.csv`.** The two come from different scripts: the tracker figure
is from `evaluate_finetuned.py`, and this table is `error_analysis.py` re-scoring
`predictions.csv`. For run2–run5 the two agree to two decimals; run1 is the only
run where they differ by 0.28 pt. The cause was not investigated. This document
uses the `comparison_summary.csv` figures throughout because every cluster count,
severity bin and confusion table below is derived from that same pipeline, so the
cross-run comparisons stay internally consistent. On a slide, quote one figure and
footnote the other.

## 1b. Severity distribution

Severity bins are by per-sample WER: `exact_match`=0, `minor`≤0.25,
`moderate`≤0.5, `major`≤1.0, `severe`>1.0 (the prediction has more
inserted/substituted words than the reference has words at all).

| Run | exact_match | minor | moderate | major | severe |
|---|---|---|---|---|---|
| **run1** | 9,355 (60.4%) | 1,850 (11.9%) | 3,174 (20.5%) | 1,075 (6.9%) | 29 (0.2%) |
| **run5** | 8,384 (52.9%) | 2,875 (18.1%) | 3,242 (20.4%) | 1,307 (8.2%) | 52 (0.3%) |
| run2 | 7,961 (51.4%) | 2,647 | 3,322 | 1,477 | 76 |
| run4 | 6,573 (42.5%) | 2,817 | 3,920 | 2,068 | 105 |
| run3 | 1,490 (9.6%) | 1,356 | 4,436 | 7,860 | 341 |

This distribution matters more than the WER average for a product decision: in
run1 and run5, roughly **four out of five wrong samples are `minor` or
`moderate`** — single-word or few-word edits, not comprehension failures. Those
are the errors a human reviewing a transcript fixes in one keystroke. `severe`
is under 0.35% in both.

## 1c. Reading of the comparison

**Ranking: run1 > run5 > run2 > run4 >> run3** — but with two caveats that the
raw ordering hides:

- **Full fine-tune beats LoRA consistently** (run1 17.36% and run5 19.06% vs.
  run2 21.01%, run4 25.99%). For this task, at this data scale, the low-rank
  adapters did not catch up.
- **run5 is not worse than run1**, despite sitting below it in the table — it
  was scored on a harder, speaker-disjoint test set. §5 is entirely about this.
- **run3 is an outlier, not a data point.** 90.4% of samples wrong and a 59.07%
  WER is not "LoRA but weaker," it is a broken run: its largest cluster (7,819
  samples) has no coherent linguistic theme, and its confusions include basic
  word errors the other runs never make (`මොකද`→`මුකද`, `විශාල`→`විෂාල`,
  `එයින්`→`එහෙන්`). It is labelled `best-epoch2`, so the most likely causes are
  (a) stopped before convergence, (b) a rank/target-module/LR mismatch, or (c) a
  checkpoint/merge mismatch between what was intended and what was evaluated.
  **run3 should not be used to argue anything about LoRA's ceiling.**

### Side note: catastrophic forgetting of English

Worth having on a slide because it drove a deployment decision. From
`finetune_tracker.csv`, the LibriSpeech test-clean check against base
Whisper-small (4.27% WER):

| Run | English WER after fine-tune | Delta vs. base | Severity |
|---|---|---|---|
| run1 (full FT) | 80.86% | **+76.59 pt** | severe |
| run2 (LoRA q,v) | 73.74% | +69.47 pt | severe |
| run4 (LoRA q,k,v,out,fc1,fc2) | 7.03% | +2.76 pt | mild |
| full-lr1e-5-cosine (full FT, lr 1e-5) | 6.06% | +1.79 pt | mild |
| run5 | not evaluated | — | — |

The pattern: run1 (full FT, lr 3e-5, linear) lost most of its English (+76pt) and
run2 (LoRA on `q,v` only) lost nearly as much (+69pt), while run4 (LoRA on all six
target modules, lr 1e-4, cosine) and the full FT at lr 1e-5 with a cosine schedule
both stayed within +3pt. These runs differ in several variables at once (LoRA vs.
full, target modules, LR, schedule), so this does not isolate a single cause; the
tracker's own notes point to the wider LoRA target set and the lower LR / cosine
schedule. After the wake word, English commands are classified by an audio
classifier that works from raw audio rather than a transcript (see
[`openWake/README.md`](../openWake/README.md)); the repo does not say whether
English forgetting motivated that choice.
run5's English check was explicitly skipped, so that cell is blank rather than
assumed.

---

# 2. The clustering strategy (how failures were grouped)

This section exists because "we clustered the errors" is the step most likely to
be challenged, and the honest answer includes what the method *cannot* do.

## 2a. The algorithm

Implemented in `final-scripts/error_analysis.py` (`write_clusters`):

1. **Select** only the wrong samples (per-sample WER > 0). run1: 6,128 of
   15,483. run5: 7,476 of 15,860.
2. **Build one string per error** by concatenating reference and prediction:
   `reference + " " + prediction`. Clustering the *pair* rather than either side
   alone is deliberate — what characterizes an error is the relationship between
   what was said and what came out, so the features need to see both.
3. **Vectorize** with TF-IDF over **character n-grams**:
   `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=5000)`.
4. **Cluster** with `KMeans(n_clusters=8, random_state=0, n_init=10)`.
5. **Report** per cluster: size, mean WER, the 8 highest-weighted n-grams at the
   cluster centroid (its "signature"), and the 5 worst-WER member examples.

## 2b. Why character n-grams, and not words or embeddings

- **No Sinhala tokenizer needed.** Word-level features would require a reliable
  Sinhala word tokenizer, and word segmentation is *itself* one of the defects
  under investigation (§3). Using word features to diagnose a word-boundary
  problem would be circular.
- **It surfaces orthographic structure directly.** Sinhala conjuncts are encoded
  as multi-codepoint sequences (`්‍ය`, `්‍ර` — consonant + virama + ZWJ + vowel).
  A 2–4 character window captures those sequences as first-class features, which
  is exactly why the ZWJ clusters fell out of the data on their own rather than
  having to be looked for.
- **`char_wb` respects word boundaries** when generating n-grams, so features
  stay interpretable as "inside a word" patterns.
- **It is cheap and fully reproducible** (`random_state=0`), which matters for a
  diagnostic that gets re-run after every training run and compared across runs.

## 2c. What this method genuinely cannot do — state this before someone asks

The clustering is **unsupervised similarity of character shapes**. It has no
notion of meaning, grammar, or root cause. Concretely:

- **Clusters are not error categories.** KMeans was told to produce 8 groups, so
  it produces 8 groups whether or not the data contains 8 phenomena. Several
  clusters in every run are grab-bags whose "top n-grams" are just frequent
  Sinhala characters (`ය`, `ම`, `්`, `න්`) — those clusters mix multiple unrelated
  causes because none of their members shared a distinctive enough signature to
  separate out.
- **A clean cluster is a coincidence of shape, not an insight.** run1's cluster 0
  is a genuine ZWJ-conjunct cluster only because those errors literally share the
  `්‍ය` byte sequence. The algorithm did not "understand" conjuncts.
- **It reports symptoms, on model output.** It looks only at test-time
  predictions, so it can tell you *where* to look, never *what to fix in the
  training data*. Every root cause in §3 and §6 came from a human reading
  clusters and then going back into the training transcripts — the clustering
  narrowed the search, it did not produce the answer.
- **`k=8` is a knob, not a finding.** Nothing was optimized (no elbow/silhouette
  search). It was chosen as "enough groups to separate the obvious themes,
  few enough to read by hand."

The right framing for a slide: **clustering is triage.** It converted 6,128
unordered failures into 8 readable piles, and reading the worst examples in each
pointed at a fixable cause (spacing) that a direct count then confirmed at 41.6%
of wrong samples. That is all it was asked to do.

---

# 3. Step 2 — clustering run1 (the 17.36% baseline)

run1's 6,128 wrong samples, grouped into 8 clusters:

| Cluster | Size | Mean WER | Signature n-grams | What it actually is |
|---|---|---|---|---|
| **1** | **2,749 (44.9%)** | 0.42 | `්`, `ව`, `ම`, `ස` (generic) | Largest and least distinctive cluster; its worst examples are spacing disagreements (only the top 5 were read) |
| 3 | 857 | 0.39 | `න්න`, `න්`, `නේ` | Verb-ending splits + colloquial endings |
| 7 | 633 | 0.38 | `ින්`, `න්`, `ම`, `ව` | Clitic-particle gluing (`ම`, `ව`) |
| 5 | 534 | 0.40 | `හැකි`, `ීම`, `ැක` | Compound-verb splits (`සිදු කිරීම`) |
| 2 | 401 | 0.38 | `්‍ර`, `ප්‍ර`, `‍ර` | **ZWJ rakaransaya conjunct** |
| 4 | 353 | 0.36 | `කිය`, `කියල` | Colloquial quotative (`කියල`/`කියලා`) |
| 6 | 323 | 0.38 | `තියෙ`, `ියෙන` | Colloquial `තියෙන`/`තියන` + spacing |
| 0 | 278 | 0.37 | `්‍ය`, `‍යා` | **ZWJ yansaya conjunct** |

## 3a. The headline finding: about 42% of wrong samples were spacing, not hearing

Measured directly, not inferred from the clusters: of run1's 6,128 wrong samples,
**2,551 (41.6%) become identical to the reference once spaces and punctuation are
removed**, and those rows carry **42.6%** of all word-level errors (5,567 of
13,062). Only 51 more (0.8%) differ solely by the ZWJ character. The model
transcribed the *right sounds* in these and disagreed only about where the word
breaks go. The largest cluster (cluster 1, 2,749 samples) has a generic
signature, and its five worst examples show the pattern:

| Reference | Prediction | Per-sample WER |
|---|---|---|
| `තුන්වැදෑරුම්ය.` | `තුන් වැදෑරුම් ය.` | **3.00** |
| `කදාවළලු,` | `කඳා වළලු` | 2.00 |
| `බොරුකීම` | `බොරු කීම,` | 2.00 |
| `ටීවිවල` | `ටීවි වල` | 2.00 |
| `එනම්ඉතිරිය` | `එනම් ඉතිරිය` | 2.00 |

Note the WER values. `තුන්වැදෑරුම්ය` is **one** word in the reference; the
prediction writes it as three. Against a 1-word reference that scores WER 3.00 —
the worst severity bucket in the entire dataset — for an utterance the model
arguably transcribed *correctly*. A handful of these inflate the corpus WER far
out of proportion to how wrong they actually are.

The same pattern runs through the other clusters:

- Cluster 5: `මෙය සිදුකිරීමට` → `මේ සිදු කිරීමට`; `ආහාර ලබාදීමට` → `ආහාර ලබා දීමට`
- Cluster 3: `හිතාගන්න බෑ` → `හිතා ගන්න බෑ.`; `මගෙන් දැනගන්න ඕනෙනං` → `මගෙන් දැන ගන්න ඕනෙ නං.`
- Cluster 0: `බුදුදහම හා මනෝවිද්‍යාව` → `බුදු දහම හා මනෝ විද්‍යාව`

## 3b. The clitic-particle sub-pattern — the clearest single signal

run1's deletion list is dominated not by content words but by **single-character
grammatical particles**:

| "Deleted" word | Count in run1 | What it is |
|---|---|---|
| `ම` | **872** | emphatic particle ("the very / itself") |
| `ව` | **407** | adverbial marker ("-ly") |
| `ද` | **289** | question / "also" particle |
| `දී` | 236 | locative ("at / in") |
| `ය` | 188 | copula ("is") |

These are not really deletions. The reference wrote the particle as a
**separate word**; the model glued it onto the preceding word. Word-level
alignment then scores that as one deletion *plus* one substitution. The
substitution list confirms it directly — **16 of run1's top 25 substitutions are
this exact glue pattern**:

| Reference → Prediction | Count | Particle glued |
|---|---|---|
| `එමෙන්` → `එමෙන්ම` | 39 | `ම` |
| `පිළිබඳ` → `පිළිබඳව` | 36 | `ව` |
| `නො` → `නොවේ` | 35 | `වේ` |
| `ඇත්තට` → `ඇත්තටම` | 29 | `ම` |
| `මෙන්` → `මෙන්ම` | 28 | `ම` |
| `මෙහෙ` → `මෙහෙම` | 22 | `ම` |
| `මුලින්` → `මුලින්ම` | 21 | `ම` |
| `කොහොම` → `කොහොමද` | 21 | `ද` |
| `පසු` → `පසුව` | 21 | `ව` |
| `යුතු` → `යුතුව` | 20 | `ව` |
| `මෙහි` → `මෙහිදී` | 20 | `දී` |
| `මොකක්` → `මොකක්ද` | 19 | `ද` |
| `හැකි` → `හැකිය` | 19 | `ය` |
| `එහි` → `එහිදී` | 18 | `දී` |
| `අනතුරු` → `අනතුරුව` | 18 | `ව` |
| `අමතර` → `අමතරව` | 17 | `ව` |

And in-context, from cluster 2:

- `ref: ප්‍රතිලෝම ව සමානුපාතික ව` → `pred: ප්‍රතිලෝමව සමානුපාතිකව`
- `ref: අතිශයින් ම ප්‍රකාශිත ය` → `pred: අතිශයින්ම ප්‍රකාශිතය`

Both predictions are arguably the *more* standard written form. The model was
being penalised for disagreeing with a transcription convention, not for
mishearing anything.

## 3c. Diagnosis drawn from step 2

1. **The dominant error class was a data-labeling inconsistency, not a model
   deficiency.** Four source corpora (OpenSLR-52 ≈ 97% of the pool, plus
   YouTube / BizBrains / Linga) were transcribed without a shared
   compounding/spacing standard. The model cannot be more consistent than its
   labels. More training on the same data would not fix this.
2. **ZWJ/conjunct encoding was inconsistent** (clusters 0 + 2 = 679 samples,
   ~11% of errors). The same visual conjunct can be encoded with or without an
   explicit ZWJ, and different keyboards/tools normalize differently.
3. **Colloquial ↔ formal register was being hedged** (`කියල`⇄`කියලා`,
   `කරන්නෙ`⇄`කරන්නේ`, `තියනවා`→`තියෙනවා`) — flagged here, but not yet acted on.
4. **A suspicion about the split itself:** v1 was not built as a speaker-disjoint
   split, so its test set may contain voices the model trained on, which would
   make these numbers look better than they should. This was a suspicion, not a
   measurement: the parquet files carry no speaker column.

Actions taken: **(1) normalize spacing/compounding conventions** across all four
corpora, and **(2) re-split speaker-disjoint** so the evaluation can no longer
reward speaker overlap. That produced `stratified_v4`.

---

# 4. Step 3 — the fix, and what `stratified_v4` actually is

`stratified_v4` = the same audio pool, with:

- **spacing/compounding normalized** across all four source corpora
- **speaker-disjoint train / validation / test** splits (no speaker appears in
  more than one split)

Documented in `final-scripts/finetune_whisper.py` and `finetuneGuide.md` as
"speaker-disjoint, spacing-normalized text." The dataset lineage:

```
stratified (v1)  ── 18k manual word/phrase QA corrections + ZWNJ fix ─→  stratified_v2
stratified_v2    ── speaker-disjoint re-split (make_speaker_split.py) ─→  stratified_v3
stratified_v3    ── (earlier attempt) ──────────────────────────────────→  stratified_v4   ← run5 trained here
stratified_v4    ── register + word-boundary normalization (this work) ─→  stratified_v5   ← published, next run
```

run5 then re-ran **run1's recipe** (full fine-tune, lr 3e-5 linear, effective
batch 64, 4 epochs, warmup 500) on `stratified_v4`, so the data is the main
deliberate difference. Two honest confounds to disclose on the slide: run5 used
effective batch 64 vs. run1's 32, and it was resumed mid-training from
`checkpoint-5778` after a `transformers`/`torch.load` (CVE-2025-32434)
incompatibility forced a pod switch.

---

# 5. Step 4/5 — run5, re-clustered: the number went up, the defect went away

run5 scored **19.06% WER** where run1 scored 17.36%. Taken at face value that
reads as a regression. It is not, and this is the most important analytical point
in the whole project.

## 5a. Why the two numbers are not comparable

| | run1 | run5 |
|---|---|---|
| Test samples | 15,483 | **15,860** |
| Test split | `stratified` (v1) | `stratified_v4` |
| Speaker-disjoint? | **No** | **Yes** |

Different test set, different size, and a stricter split. v1 was not built as a
speaker-disjoint split, so run1's test set may share speakers with its training
data, which would flatter its 17.36%. That effect was not measured, so it is a
likely contributor rather than a quantified one. run5's 19.06% is measured on a
speaker-disjoint split, so it reflects **voices the model had not heard**. A
higher number on a stricter test is not necessarily a worse model.

The error-type breakdown shows exactly this trade:

| | Substitutions | Deletions | Insertions |
|---|---|---|---|
| run1 | 11.47% | **4.68%** | 1.21% |
| run5 | 14.67% | **2.33%** | 2.07% |

- **Deletions halved** (4.68% → 2.33%). Consistent with the spacing fix landing,
  since the glue pattern shows up as deletions.
- **Substitutions rose** (11.47% → 14.67%). Plausibly the cost of unseen speakers
  (acoustic confusions on new voices), though this run did not isolate that.

## 5b. Evidence the spacing fix worked (before/after, not a controlled ablation)

The direct measurement from §3a, repeated on run5: wrong samples that differ from
the reference only by spaces/punctuation fell from **2,551 (41.6% of wrong)** to
**1,013 (13.6%)**, and the word-level errors they carry fell from **5,567 to
2,147** (42.6% → 15.0% of all word errors), even though the run5 test set is
slightly larger.

The clitic-particle deletions from §3b, before and after:

| "Deleted" particle | run1 | run5 | Change |
|---|---|---|---|
| `ම` | 872 | **80** | **−91%** |
| `ව` | 407 | **48** | **−88%** |
| `ද` | 289 | **73** | −75% |
| `දී` | 236 | **11** | **−95%** |
| `ය` | 188 | **106** | −44% |
| **Top-5 total** | **1,992** | **318** | **−84%** |

And the glue-pattern substitutions, which were 16 of run1's top 25:

| | Glue-pattern entries in top-25 substitutions |
|---|---|
| run1 | **16 / 25** (`එමෙන්`→`එමෙන්ම`, `පිළිබඳ`→`පිළිබඳව`, …) |
| run5 | **1 / 25** (only `යුතු`→`යුතුය`) |

The spacing error class that was 41.6% of run1's wrong samples is down to 13.6%
in run5, and the particle-gluing pattern has left the top failure modes. **This is
the result to put on the slide**, framed as before/after evidence rather than proof
of cause: run1 and run5 differ in more than the data (the test split, effective
batch 64 vs. 32, and a mid-run resume), so this shows the defect largely went away
after the fix, not that the spacing fix alone was responsible. The headline
average moved the other way, plausibly because of the stricter test split.

## 5c. run5's clusters — what is left

| Cluster | Size | Mean WER | Signature | What it is |
|---|---|---|---|---|
| 0 | 2,656 | 0.41 | generic (`ස`, `ව`, `ය`) | Mixed: register + rare words + proper nouns |
| 7 | 1,255 | 0.39 | `ක්`, `්` | Mixed residual |
| 2 | 1,038 | 0.39 | `න්න`, `ෙන` | Verb endings, register |
| 6 | 993 | 0.37 | `න්`, `ින්` | Mixed residual |
| **5** | **652** | 0.37 | `්‍ර`, `ප්‍ර` | **ZWJ rakaransaya — still present** |
| 1 | 472 | 0.38 | `කිය`, `කියල` | **Register (`කියල`/`කියලා`)** |
| 3 | 280 | 0.36 | `හැකි`, `නිසා` | Compound-verb splits |
| **4** | **130** | 0.37 | `වහන්සේ` | **Honorific/religious register** |

Still alive, in priority order:

**(1) Register inconsistency is now the #1 residual — and it is provably a label
defect.** The tell is that the *same pair errs in both directions*:

| Pair | A → B | B → A |
|---|---|---|
| `කියල` / `කියලා` | 91 | 23 |
| `එය` / `ඒ` | 20 | 18 |
| `සමග` / `සමඟ` | 16 | 13 |

A model that mishears a word does not mishear it symmetrically both ways. A
model trained on a corpus that spells the same word two ways, with no consistent
rule, produces exactly this. Corpus-wide counts confirm it: `සමග` appears 353
times and `සමඟ` 345 — a 51/49 coin flip, so the model gets no signal at all
about which to emit.

**(2) ZWJ conjuncts survived the v4 pass** (cluster 5, 652 samples). Example from
run5's own output: `ref: ක්‍රියාකළ යුතුයි.` → `pred: ක්‍රියා කළ හේතුවෙයි.` — the
compound `ක්‍රියාකළ` split at the conjunct. Whatever spacing normalization v4
applied did not cover conjunct-boundary compounds.

**(3) Honorific/religious register** (cluster 4, 130 samples) — e.g.
`මහාකස්සපයන්` → `මහා කශ්ෂපයන්`. Low volume, high per-sample WER. This is a
data-*volume* problem (sermon/Dhamma content is a small slice of the pool), not a
labeling one, so text normalization will not touch it.

**(4) Rare proper nouns and abbreviations** — `එස්.ඩබ්ලිව්.ආර්.ඩී.බණ්ඩාරනායක` →
`එස් ඩබ්ලිව් ආර්ලිය වණ්ඩාරනායක`. Genuine acoustic/OOV difficulty. Lowest
priority.

## 5d. One more check: is the test set leaking?

Because §5a rests on "v4 is speaker-disjoint," that claim was verified rather
than trusted. 16,079 sentences (~10% of the corpus) have **identical text** in
both train and test/validation — which looks alarming. Every row's audio was
then SHA-256 hashed and compared across splits: **zero** cases of the same audio
file appearing in more than one split. So it is a fixed prompt list read by
different speakers (normal for read-speech corpora), not audio leakage.
Speaker-disjointness holds, and run5's 19.06% is not inflated by memorization.
Full method in [`run5/NORMALIZATION_APPROACH.md`](run5/NORMALIZATION_APPROACH.md) §7.

---

# 6. Step 6 — what was built in response, and what is still open

The §5c diagnosis drove a further normalization pass, now published as
**`stratified_v5`** (`Yohan2003/whisper-sl-data`, `data/stratified_v5/`):

- **407 register-spelling pairs** unified (`කියල`→`කියලා`, `සමග`→`සමඟ`,
  `තුල`→`තුළ`, …)
- **60 word-boundary/ZWJ-conjunct pairs** glued (`ක්‍රියා කළ`→`ක්‍රියාකළ`,
  `විශ්ව විද්‍යාලයේ`→`විශ්වවිද්‍යාලයේ`, …)
- **13,120 of 154,828 rows changed (~8.5%)**; audio untouched; `stratified_v4`
  left in place so run5 stays reproducible

The full method — including the 12 false-positive pairs that had to be caught by
hand (e.g. `ගන්න` "to take" vs. `ගන්නා` "the one who takes" are different words,
not spelling variants) — is in
[`run5/NORMALIZATION_APPROACH.md`](run5/NORMALIZATION_APPROACH.md). That document
is the one to cite for the "how I approached reducing WER" part of a
presentation; this one is the evidence base.

Still open:
- The next fine-tune on `stratified_v5` has not been run, so v5's effect on WER
  is **unmeasured**. Nothing in this document should be read as claiming v5
  improves WER — only that it removes a defect that provably drove run5's errors.
- Honorific register (§5c.3) and rare proper nouns (§5c.4) need *data*, not text
  fixes.
- A dictionary/spellcheck pass is missing. This method only finds words spelled
  inconsistently *between two forms that both exist*; it is blind to words
  misspelled the same wrong way everywhere — e.g. `ප්‍රථිපල` and `ප්‍රතිපල` are
  both typos of `ප්‍රතිඵල`, which appears nowhere in the corpus.

# 7. Practical takeaways

1. **Full fine-tune over LoRA** for this task and data scale — consistently, across
   every comparison made here.
2. **Judge an intervention by the error class it targeted, not by the headline
   average.** run5's WER rose and the fix still worked (−84% on the targeted
   class). The two facts have separate causes.
3. **Never compare WER across different test splits.** run1 vs. run5 is the
   cautionary example; the split changed underneath the number.
4. **Bidirectional confusion pairs are the signature of a label defect,** not an
   acoustic one. They are the cheapest high-confidence thing to look for in a
   confusion table.
5. **Fix data conventions before buying more compute.** The largest single error
   class in the baseline (41.6% of wrong samples differed only by spacing) was a
   transcription-convention disagreement that additional training on the same
   labels would not be expected to resolve.
6. **For LoRA deployment**, add decoding-time repetition guards
   (`no_repeat_ngram_size=3`): the degenerate-loop failure mode
   (`මෙම මෙම මෙම…` up to WER 7.60) appears in the LoRA runs (run2, run4) and not
   in the full fine-tunes.
7. **Aggressive fine-tuning can wreck English** — +76.59pt English WER on run1 and
   +69.47pt on run2, versus +2 to +3pt for the gentler configurations. English
   commands are classified by a separate raw-audio classifier
   ([`openWake/README.md`](../openWake/README.md)); the repo doesn't say whether
   forgetting motivated that.
