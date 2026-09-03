# Final datasets

Two source corpora, QA'd and merged, then partitioned into two independent
split configurations for training. All parquet files share the same schema:

| column          | type   | notes                                   |
|-----------------|--------|------------------------------------------|
| `audio`         | binary | WAV bytes (PCM 16-bit), ready to decode  |
| `source_dataset`| string | `openslr`, `youtube`, `bizbrains`, `linga` |
| `text`          | string | Sinhala transcript (code-mixed English kept) |

## `final_dataset/` — the two source corpora

### `final_dataset_openslr_qa.parquet` — 149,926 rows

OpenSLR-52 Sinhala, cleaned end to end:

1. VAD-trimmed + deduplicated (`data/raw/openslr_52/processed/openslr_52_clean/`)
2. Word/phrase corrections from the SinSpeech corpus-cleaning review
   (`openslrTranscription/Corpus-Cleaning/`) applied to the transcript text
3. 265 utterances SinSpeech flagged as suspicious after listening to the audio
   were manually reviewed and removed
4. 809 rows had a stray zero-width joiner (ZWJ) stripped from the text —
   ZWJ is only valid immediately after a virama (`්`) and before ර/ය/ෂ
   (rakaranshaya/yansaya/ksha conjuncts); everything else was an artifact

### `final_collection_qa.parquet` — 4,902 rows

YouTube + BizBrains + Linga, combined and cleaned:

1. Combined, audio-trimmed, and transcript-cleaned
   (see `dse-project/model-development/notebooks/`)
2. Deduplicated against OpenSLR by audio content hash — rows whose audio was a
   byte-for-byte duplicate of an OpenSLR recording were dropped (~67% of the
   pre-dedup 15,002 rows; the same clip surfaced under two source labels)
3. Hidden/invisible characters cleaned
4. Manual QA review in progress — flagged rows checked against
   `data/processed/collection_qa_manual_review_log.csv`; 2 rows deleted,
   2 rows edited so far, review ongoing (see caveat below)
5. 2 rows had a stray ZWJ stripped, same rule as OpenSLR

Combined total: **154,828 rows** across both corpora.

## `final_split_dataset/` — the two split configurations

Built by `model-development/scripts/split_final_datasets.py` from the two files
above. Every row appears in exactly one split *within* each configuration, but
a given row can appear in both configurations (e.g. `stratified/train` and
`held_out/train`) since each is an independent partition of the same pool.

### `stratified/` — primary split, train your shipped model on this

80 / 10 / 10 train / validation / test. Every split gets the same proportion
from each of the four sources as that source's share of the combined pool, so
train/val/test all see a representative, blended mix of every domain.

| source     | train   | validation | test   |
|------------|--------:|-----------:|-------:|
| openslr    | 119,940 |     14,993 | 14,993 |
| youtube    |   2,452 |        306 |    307 |
| bizbrains  |     782 |         98 |     97 |
| linga      |     688 |         86 |     86 |
| **total**  | **123,862** | **15,483** | **15,483** |

### `held_out/` — secondary, cross-domain robustness diagnostic

`held_out_test` is carved *only* from the collection domain (youtube/bizbrains/
linga) — OpenSLR contributes nothing to it, and it's fully excluded from
train/validation in this configuration. The rest is stratified into
train/validation the same way as above. This measures how well a model
trained mostly on OpenSLR generalizes to a domain slice it never saw during
training — not the split to train your shipped model on.

| source     | train   | validation | held_out_test |
|------------|--------:|-----------:|---------------:|
| openslr    | 134,933 |     14,993 |              0 |
| youtube    |   2,207 |        245 |            613 |
| bizbrains  |     703 |         78 |            196 |
| linga      |     619 |         69 |            172 |
| **total**  | **138,462** | **15,385** | **981** |

Fractions used: 80/10/10 stratified split; `held_out_test` = 20% of the
collection-domain rows, remaining pool split 90/10 train/validation.

## Regenerating

```
python3 model-development/scripts/split_final_datasets.py
```

Re-run this any time either source file changes (e.g. after applying more
decisions from an ongoing QA review) to bring both split configurations back
in sync. Flags let you change the ratios — see the script's `--help`.

## Caveat: collection QA is not finished

`final_collection_qa.parquet` reflects only what's been reviewed so far in
`dse-project/model-development/notebooks/transcriptPreprocess/review_flagged_collection_rows.ipynb`
(candidates in `data/processed/collection_qa_flagged_candidates.csv`, 268
total, most not yet reviewed). OpenSLR's QA is complete. As more collection
rows get reviewed, re-apply the log and re-run the split script above.
