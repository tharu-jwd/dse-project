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
long review campaign. Keyboard shortcuts 1–4 select and save correct, edited,
bad audio, or uncertain. Changing the radio decision or transcript also saves
automatically. Exact duplicates are assigned from decoded-audio hashes and are
not a reviewer memory task. Space plays or replays the current audio; Left and
Right navigate between rows.

After every gold candidate has a decision, lock a new immutable version. This
command fails on missing candidates, unknown IDs, invalid accepted transcripts,
or an existing non-empty output directory:

```bash
PYTHONPATH=src python scripts/finalize_dataset.py \
  --manifest data/versions/v1/manifest.parquet \
  --adjudications reports/review/gold-v1-adjudications.jsonl \
  --output-dir data/versions/v2
```

For optional GPT spelling/format suggestions, export ID-aligned UTF-8 TSV
batches. Suggestions are never applied as ground truth without audio review:

```bash
PYTHONPATH=src python scripts/export_transcripts_for_gpt.py \
  --queue reports/review/gold-v1-candidates.parquet \
  --output-dir reports/review/gpt-suggestions/input
```

Returned batches are structurally validated and classified with
`scripts/analyze_gpt_suggestions.py`. The UI discovers the resulting
`suggestions.parquet` automatically and offers changed text for one-click
acceptance only after listening to the audio.

An explicitly owner-approved suggestion pass can be converted to a complete
overlay with `scripts/apply_gpt_suggestions.py`. Existing native audio reviews
override text-only suggestions. See `REVIEW_PROVENANCE.md`; never describe a
text-only suggestion as audio-verified.

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

Boundary-silence analysis of the 2,000 v1 gold candidates uses 20 ms frames at
-40 dBFS. Median leading/trailing silence is 1.34/0.98 seconds; 1,815 clips have
over 40% combined boundary silence. This is not a reviewer rejection criterion.
Raw clips remain immutable; conservative trimming with retained margins must be
evaluated as an explicit preprocessing ablation.

## Current frozen dataset

Dataset v3 supersedes v1 and v2 for new experiments. It contains 182,665 train,
1,000 validation, 999 test, 626 heldout-unused, and 3 excluded rows. Its
fingerprint is
`5cee7c7b91f5d7cab5ce10bab2ba85f6b18d49e1ab24fbeb50751d0fc374c31a`.
Train, validation, and test have zero speaker overlap.

V2 applied the complete owner-approved transcript overlay. V3 adds the
declared English correction in `configs/data/owner-text-overrides-v3.json` and
five owner edits saved after v2 was frozen. The reproducible sequence is:

```bash
PYTHONPATH=src python scripts/apply_text_overrides.py \
  --adjudications reports/review/gold-v1-owner-approved.jsonl \
  --overrides configs/data/owner-text-overrides-v3.json \
  --output reports/review/gold-v1-owner-approved-v3.jsonl

PYTHONPATH=src python scripts/finalize_dataset.py \
  --manifest data/versions/v1/manifest.parquet \
  --adjudications reports/review/gold-v1-owner-approved-v3.jsonl \
  --output-dir data/versions/v3
```

The full-corpus silence and clipping audit is documented in `AUDIO_AUDIT.md`.
Fifty risk-stratified crop proposals were reviewed as safe, but a matched local
training A/B showed worse early validation metrics and identical model FLOPs
with cropping. Consequently v3 retains immutable source audio and the baseline
training configuration leaves dynamic cropping disabled.

## Dataset v4 evaluation-reference review

Dataset v3 remains frozen. Because 295 validation/test references contain
text-only revisions that were not verified against audio, v4 restores the
conservatively normalized OpenSLR transcript as the neutral starting reference.
A self-contained native-listening queue contains all 295 disputed rows plus 100
deterministically selected unchanged controls:

```bash
PYTHONPATH=src python scripts/build_v4_review_queue.py \
  --manifest data/versions/v3/manifest.parquet \
  --output reports/review/v4-evaluation-queue.parquet \
  --controls 100

PYTHONPATH=src streamlit run scripts/review_app.py --server.port 8503 -- \
  --queue reports/review/v4-evaluation-queue.parquet \
  --output reports/review/v4-evaluation-adjudications.jsonl \
  --suggestions reports/review/v4-control-gpt/analysis/suggestions.parquet
```

In this queue, `Correct` means the displayed normalized OpenSLR transcript
matches the audio. For disputed rows the UI also shows the previous v3 revision;
use it only after listening confirms it. Otherwise edit the verified transcript.
Mark unusable audio as `Bad audio` and unresolved speech as `Uncertain`.

Only after all 395 rows are reviewed may v4 be frozen:

```bash
PYTHONPATH=src python scripts/finalize_dataset_v4.py \
  --manifest data/versions/v3/manifest.parquet \
  --queue reports/review/v4-evaluation-queue.parquet \
  --adjudications reports/review/v4-evaluation-adjudications.jsonl \
  --output-dir data/versions/v4
```

The finalizer refuses partial or extra decisions. It resets every validation and
test reference to the normalized original, applies only audio-reviewed edits,
and excludes reviewed bad/uncertain rows. Training rows, audio, and speaker
assignments remain unchanged.
