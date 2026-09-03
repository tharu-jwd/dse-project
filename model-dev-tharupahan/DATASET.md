# Dataset Operations

This is the operational companion to `PLAN.md`. Raw snapshots are immutable,
ignored by Git, and identified by the revisions in `configs/data/sources.json`.

## Source policy

| Source | Role | License status |
|---|---|---|
| Official OpenSLR-52 | Primary corpus | CC BY-SA 4.0 declared upstream |
| SPEAK-ASR YouTube Sinhala | Candidate domain/code-switch corpus | Unresolved; private audit only until clarified |
| SPEAK-ASR BizBrains | Candidate domain/code-switch corpus | Unresolved; private audit only until clarified |
| Lingalingeswaran JSON v1 | Provenance comparison only | Unresolved and suspected OpenSLR overlap |
| Team GCS `finalData` | Processed comparison artifact | Unavailable while owning billing account is delinquent |

Public availability is not treated as permission to train or redistribute. A
source with unresolved licensing cannot enter a releasable training dataset.

## Local layout

```text
data/
├── raw/             # immutable upstream snapshots
├── indexes/         # lightweight canonical indexes; raw audio remains unchanged
└── versions/        # frozen derived manifests and split definitions
reports/
├── sources/         # file checksums and source fingerprints
├── dataset-audit/   # automatic quality and leakage reports
└── review/          # self-contained review queues and adjudication overlays
```

## Reproducible source revisions

The pinned Hugging Face revisions are in `configs/data/sources.json`. Download
only Parquet data and the card into their corresponding raw directory:

```bash
hf download REPOSITORY \
  --repo-type dataset \
  --revision COMMIT_SHA \
  --include 'data/*.parquet' \
  --include README.md \
  --local-dir data/raw/SOURCE
```

Download OpenSLR-52 from its official SLR52 mirror. The downloader resumes
partial transfers, fetches four shards concurrently, validates every ZIP member,
blocks path traversal, extracts locally, and removes archives after success:

```bash
PYTHONPATH=src python scripts/download_openslr52.py
PYTHONPATH=src python scripts/index_openslr52.py
```

Fingerprint every completed snapshot:

```bash
PYTHONPATH=src python scripts/inventory_sources.py
```

Do not fingerprint active `.part` files; the inventory deliberately ignores
them.

## Automatic audit

Pass every upstream Parquet shard with its original split name to
`scripts/prepare_data.py`. Use `--allow-invalid` only for the exploratory run:
it still records every violation and does not authorize training.

The audit records encoded and decoded-audio SHA-256 hashes. Decoded hashes find
identical waveforms stored in different containers. It reports audio hours,
duration distribution, transcript flags, duplicates, code-switching, and
cross-split audio/sample/speaker leakage.

After all individual audits finish, combine their manifests and measure exact
cross-source overlap before creating any split:

```bash
PYTHONPATH=src python scripts/combine_manifests.py \
  --manifest reports/dataset-audit/openslr52-upstream/manifest.parquet \
  --manifest reports/dataset-audit/youtube-upstream/manifest.parquet \
  --manifest reports/dataset-audit/bizbrains-upstream/manifest.parquet \
  --manifest reports/dataset-audit/linga-upstream/manifest.parquet \
  --output-dir reports/dataset-audit/combined
```

## Review queue and application

Build the licensed v1 manifest and deterministic speaker-disjoint pools first.
The validation/test rows are candidates—not gold data—until a native reviewer
accepts or corrects them. All other rows from those speakers stay out of train:

```bash
PYTHONPATH=src python scripts/build_dataset_v1.py \
  --manifest reports/dataset-audit/openslr52-upstream/manifest.parquet \
  --output-dir data/versions/v1

PYTHONPATH=src python scripts/build_gold_review_queue.py \
  --manifest data/versions/v1/manifest.parquet \
  --output reports/review/gold-v1-candidates.parquet
```

Build a deterministic, self-contained queue from an audit manifest:

```bash
PYTHONPATH=src python scripts/build_review_queue.py \
  --manifest reports/dataset-audit/combined/manifest.parquet \
  --output reports/review/initial.parquet \
  --quota 100
```

The queue contains audio bytes so review does not depend on the source files
remaining mounted. Start the local UI:

```bash
PYTHONPATH=src streamlit run scripts/review_app.py -- \
  --queue reports/review/initial.parquet \
  --output reports/review/adjudications-v1.jsonl
```

The output is an atomic, resumable correction overlay keyed by stable sample
ID. Raw source rows are never edited. Back up the adjudication file during a
long review campaign.

## Verified findings (2026-09-03 snapshots)

- Official OpenSLR-52 contains 185,293 rows and 224.50 hours. Two clips exceed
  the current 30-second limit; 185,291 rows pass blocking validation. It has 478
  speaker identifiers, no exact decoded-audio duplicates, and no published
  train/validation/test split in this snapshot.
- Linga JSON-v1 contains 11,357 rows and 13.77 hours. Every decoded waveform is
  already present in OpenSLR-52, so it contributes zero independent audio and
  is excluded from v1 training.
- YouTube contains 4,037 rows and 9.11 hours. Its published split leaks 57 video
  recording groups across split boundaries, so those splits cannot be used for
  evaluation.
- BizBrains contains 979 rows and 2.58 hours. It has two internal exact-audio
  duplicate groups, including one train/test leak. In addition, 959 decoded
  waveforms (98.16% of its unique waveforms) already occur in YouTube.
- The YouTube, BizBrains, and Linga dataset cards do not declare usable license
  terms. They remain provenance/audit inputs and do not enter licensed v1.

The generated evidence is in `reports/dataset-audit/*` and
`reports/sources/inventory.json`. The combined cross-source fingerprint is
`465c3303c4cb646ef16b520af477c47964ee0d5de2d4e30369c3f50f291ce632`.
The deterministic v1 split fingerprint is
`410fe5eddffb7d597398fa46c07beaf1d77c6e3a986aaa92fecb48ebaac86e83`.
