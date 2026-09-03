# Dataset Operations

This is the operational companion to `PLAN.md`. Raw snapshots are immutable,
ignored by Git, and identified by the revisions in `configs/data/sources.json`.

## Source policy

| Source | Role | License status |
|---|---|---|
| Official OpenSLR-52 | Primary corpus | CC BY-SA 4.0 declared upstream |
| SPEAK-ASR YouTube Sinhala | Candidate domain/code-switch corpus | Unresolved; private audit only until clarified |
| SPEAK-ASR BizBrains | Candidate domain/code-switch corpus | Unresolved; private audit only until clarified |
| Lingalingeswaran v4 | Provenance comparison only | Unresolved and suspected OpenSLR overlap |
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

Download OpenSLR-52 from its official SLR52 mirrors, including `LICENSE`,
`utt_spk_text.tsv`, and all sixteen `asr_sinhala_[0-f].zip` shards. Extract to
`data/raw/openslr52` and build its lightweight index:

```bash
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

## Review queue and application

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

## First verified finding

The upstream BizBrains snapshot contains 979 valid decodable rows and 2.58
hours of audio. Its published splits contain two exact decoded-audio duplicate
groups; one group crosses train and test. Therefore its upstream test split is
not acceptable as a locked evaluation set. This finding is reproducible from
`reports/dataset-audit/bizbrains-upstream`, which is generated and intentionally
not committed.
