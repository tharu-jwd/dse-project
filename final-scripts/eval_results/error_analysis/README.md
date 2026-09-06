# Error Analysis — Whisper Small Sinhala (run1–run4)

This folder holds the output of `error_analysis.py` run against the test-set predictions of all four fine-tuning runs. For each run it groups every wrong sample into 8 clusters (TF-IDF over **character n-grams** + KMeans — chosen so it works directly on Sinhala script without needing a word tokenizer), buckets every sample by severity, and tallies the most common substitution / deletion / insertion words. This document summarizes what those outputs actually show and why.

Files per run: `<run>_clusters.txt` (themed failure groups), `<run>_errors_by_severity.csv` (every wrong sample, worst first), `<run>_confusions.txt` (top substitution/deletion/insertion words).

## 1. Headline numbers (15,483 test samples per run)

| Run | Type | Wrong samples | Mean WER | exact_match | minor | moderate | major | severe |
|---|---|---|---|---|---|---|---|---|
| **run1_full** | Full fine-tune | 6,128 (39.6%) | **0.158** | 9,355 (60.4%) | 1,850 | 3,174 | 1,075 | 29 |
| **run2_amd_lora** | LoRA (AMD hw) | 7,522 (48.6%) | 0.199 | 7,961 (51.4%) | 2,647 | 3,322 | 1,477 | 76 |
| **run4_lora** | LoRA | 8,910 (57.5%) | 0.250 | 6,573 (42.5%) | 2,817 | 3,920 | 2,068 | 105 |
| **run3_lora** | LoRA (best-epoch2) | 13,993 (90.4%) | **0.575** | 1,490 (9.6%) | 1,356 | 4,436 | 7,860 | 341 |

Severity bins (by per-sample WER): `exact_match`=0, `minor`≤0.25, `moderate`≤0.5, `major`≤1.0, `severe`>1.0 (more inserted/substituted words than the reference has).

**Ranking: run1 > run2 > run4 >> run3.** The full fine-tune generalizes best; both merged/adapter LoRA runs trail it by a moderate margin; run3 is a clear outlier and should not be used as-is (see §4).

## 2. What the clusters actually contain

Across every run, the 8 clusters are not random — the same handful of *themes* keep reappearing, just at different rates:

1. **Conjunct-consonant / ligature clusters** — keyed on n-grams like `්‍ය`, `‍ය`, `්‍ර`, `‍ර` (the yansaya and rakaransaya conjuncts, which Unicode Sinhala renders with a zero-width joiner). e.g. run1 cluster 0 (278 samples) and cluster 2 (401 samples) are almost entirely this. Predictions get the base consonant right but drop or restructure the conjunct: `සත්‍යයද` → `සත්‍යය ද`, `මනෝවිද්‍යාව` → `මනෝ විද්‍යාව`.
2. **Word-boundary / compounding clusters** — the single largest cluster in nearly every run (run1 cluster 1: 2,749 samples; run2 cluster 0: 3,146; run3 cluster 2: 7,819; run4 cluster 3: 3,220). The model splits words that are written joined in the reference, or joins words that are written apart: `තුන්වැදෑරුම්ය` → `තුන් වැදෑරුම් ය`, `කදාවළලු` → `කඳා වළලු`, `ටීවිවල` → `ටීවි වල`.
3. **Colloquial-ending clusters** (`කිය`, `කියල`, `ගන්න`, `න්න`) — spoken-register verb/quotative endings where the model normalizes toward the "book" form: `කියල` → `කියලා`, `කරන්නෙ` → `කරන්නේ`, `තියනවා` → `තියෙනවා`.
4. **Honorific / religious-register cluster** (`වහන්සේ`, `න්සේ`) — present as its own cluster in run2 (123), run3 (259), run4 (151): `භික්‍ෂූන්වහන්සේ` → `භික්ෂූන් වහන්සේ`, `උන්වන්සේගේ` → `උන්වහන්සේගේ`.
5. **Degenerate repetition ("hallucination loop") cluster** — the single worst individual samples in run2 and run4 (WER up to 7.60) are the decoder getting stuck repeating one word/phrase dozens of times (`මෙම මෙම මෙම...` repeated ~40 times for a 9-word reference). Not present as a distinct cluster in run1.

## 3. Root-cause analysis

### a) Formal vs. colloquial orthography mismatch (biggest single driver)
The #1 substitution pattern in **every run** is a spoken-register word being "corrected" to (or from) its formal/written equivalent: `කියල⇄කියලා`, `කරන්නෙ⇄කරන්නේ`, `එමෙන්⇄එමෙන්ම`, `කථාව⇄කතාව`, `මොකද⇄මුකද`. Sinhala has a large gap between spoken and written registers, and the training corpora mix registers unevenly — OpenSLR-52 (97% of the training pool) is read/formal speech, while YouTube/BizBrains/Linga are conversational. The model ends up hedging between conventions rather than reliably reproducing whichever one is in front of it. **This alone likely explains a large share of "minor" and "moderate" severity errors** in every run, since these are single-word edits, not comprehension failures.

### b) Inconsistent word-boundary/compounding conventions across source datasets
The largest cluster in 3 of 4 runs is pure spacing: the model and the reference disagree on whether a compound is one word or two (`ටීවිවල` vs `ටීවි වල`, `බොරුකීම` vs `බොරු කීම`). This is consistent with **the four source corpora not sharing a single transcription/compounding standard** — a scripted correction pass on OpenSLR vs. ad-hoc human transcription on YouTube/collected data would produce exactly this kind of disagreement, and the model can't do better than the labels it was trained on. This is a **data-labeling consistency issue**, not a model-capacity issue — it will not go away with more training on the same data.

### c) Zero-width-joiner (ZWJ) / conjunct-consonant rendering inconsistency
Sinhala conjuncts (යන්සය, රකාරාංශය) can be encoded in Unicode with or without an explicit ZWJ, and different tools/keyboards normalize them differently. The recurring `්‍ය` / `‍ය` / `්‍ර` / `‍ර` clusters in run1 (679 of 6,128 wrong samples, ~11%) point to the training transcripts not being normalized to one encoding, so the model learned an inconsistent mapping for these characters specifically. **Recommendation: run a Unicode NFC/ZWJ-normalization pass over all transcripts before the next training run** — this is a cheap, one-time data fix that should directly reduce this cluster.

### d) Underrepresented religious/formal vocabulary
The `වහන්සේ` (Buddhist monastic honorific) cluster shows up across LoRA runs with the model dropping or garbling the honorific specifically (`උන්වන්සේගේ` → `උන්වහන්සේගේ`, i.e. missing a syllable). This is very likely sermon/Dhamma-talk content from the YouTube slice (only ~2% of training data) — too little exposure for the model to lock in this specific, low-frequency but structurally distinct vocabulary.

### e) LoRA capacity/training-length limits → degenerate repetition
The exact-repeated-token failure mode (`මෙම මෙම මෙම...`) only shows up prominently in **LoRA** runs (run2, run4), not the full fine-tune (run1). This is a classic symptom of a decoder that hasn't been adapted enough to confidently terminate generation on out-of-distribution or acoustically ambiguous input (background noise, very short utterances, rare words) — it falls back to repeating its highest-probability token instead of stopping. A full fine-tune updates all weights and is more robust to this; a low-rank adapter has less capacity to correct the base model's decoding behavior on hard inputs. **Recommendation: add a repetition penalty / no_repeat_ngram_size at generation time for the LoRA checkpoints**, and consider a higher LoRA rank if this persists after that.

### f) run3 is a clear outlier — likely an undertrained/wrong checkpoint, not a data problem
run3's error profile isn't just "a bit worse" — 90% of samples are wrong (vs. ~40–58% for the others) and the *shape* of its errors is different: its largest cluster (7,819 samples, avg WER 0.64) has no coherent linguistic theme, and its confusions include basic word confusions the other runs don't make (`මොකද`→`මුකද`, `විශාල`→`විෂාල`, `එයින්`→`එහෙන්`) alongside the same orthographic issues seen elsewhere. Since this run is labeled "best-epoch2", it suggests either:
  - training was stopped too early (epoch 2) and the adapter hadn't converged, or
  - the LoRA config for this run (rank/target modules/learning rate) was mismatched to the task, or
  - there's a checkpoint/merge mismatch between what was evaluated and what was intended to be evaluated.

  **Recommendation: re-check run3's training logs/config before drawing any conclusions from it, and re-run evaluation on its later checkpoints (`checkpoint-3871`, `checkpoint-7742`) if they exist** — `best-epoch2` may simply have been selected on too little training.

## 4. Practical takeaways

- **Use run1 (full fine-tune)** as the primary model; it has the lowest error rate and the cleanest error profile.
- **Don't ship run3 as-is** — its errors aren't representative of the LoRA approach's ceiling, they're representative of an undertrained/misconfigured run.
- **Before the next training run**, normalize transcript orthography: (1) pick one convention for compounding/spacing and re-run it across all four source datasets, (2) NFC-normalize all Sinhala text to fix ZWJ/conjunct inconsistencies. Both are pure data fixes and should reduce error rates across all future runs regardless of model/LoRA choice.
- **For LoRA deployment**, add decoding-time repetition guards (`no_repeat_ngram_size=3` or similar) to eliminate the degenerate-loop failure mode seen on hard inputs.
- Religious/formal-register vocabulary (`වහන්සේ` and similar) would benefit from targeted additional data if that domain matters for the product.
