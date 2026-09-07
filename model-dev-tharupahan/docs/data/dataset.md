# Dataset Operations

This is the operational companion to [the project plan](../project/plan.md).
Raw snapshots are immutable,
ignored by Git, and identified by the revisions in `configs/data/sources.json`.

## Source policy

| Source | Role | License status |
|---|---|---|
| Official OpenSLR-52 | Primary corpus (train + validation only, see v5 below) | CC BY-SA 4.0 declared upstream |
| SPEAK-ASR YouTube Sinhala | Test set (eval-only, never train) | Unresolved upstream; owner-approved for private eval-only use (2026-09-07) |
| Path Nirvana Sinhala TTS | Test set (eval-only, never train) | GPL-3; voices donated for non-obscene speech generation -- read as covering private transcription-accuracy evaluation, not redistribution/resynthesis |
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
PYTHONPATH=src python scripts/data/download_openslr52.py
PYTHONPATH=src python scripts/data/index_openslr52.py
```

Fingerprint every completed snapshot:

```bash
PYTHONPATH=src python scripts/data/inventory_sources.py
```

Do not fingerprint active `.part` files; the inventory deliberately ignores
them.

## Automatic audit

Pass every upstream Parquet shard with its original split name to
`scripts/data/prepare_data.py`. Use `--allow-invalid` only for the exploratory run:
it still records every violation and does not authorize training.

The audit records encoded and decoded-audio SHA-256 hashes. Decoded hashes find
identical waveforms stored in different containers. It reports audio hours,
duration distribution, transcript flags, duplicates, code-switching, and
cross-split audio/sample/speaker leakage.

After all individual audits finish, combine their manifests and measure exact
cross-source overlap before creating any split:

```bash
PYTHONPATH=src python scripts/data/combine_manifests.py \
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
PYTHONPATH=src python scripts/data/build_dataset_v1.py \
  --manifest reports/dataset-audit/openslr52-upstream/manifest.parquet \
  --output-dir data/versions/v1

PYTHONPATH=src python scripts/review/build_gold_review_queue.py \
  --manifest data/versions/v1/manifest.parquet \
  --output reports/review/gold-v1-candidates.parquet
```

Build a deterministic, self-contained queue from an audit manifest:

```bash
PYTHONPATH=src python scripts/review/build_review_queue.py \
  --manifest reports/dataset-audit/combined/manifest.parquet \
  --output reports/review/initial.parquet \
  --quota 100
```

The queue contains audio bytes so review does not depend on the source files
remaining mounted. Start the local UI:

```bash
PYTHONPATH=src streamlit run scripts/review/review_app.py -- \
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
PYTHONPATH=src python scripts/data/finalize_dataset.py \
  --manifest data/versions/v1/manifest.parquet \
  --adjudications reports/review/gold-v1-adjudications.jsonl \
  --output-dir data/versions/v2
```

For optional GPT spelling/format suggestions, export ID-aligned UTF-8 TSV
batches. Suggestions are never applied as ground truth without audio review:

```bash
PYTHONPATH=src python scripts/review/export_transcripts_for_gpt.py \
  --queue reports/review/gold-v1-candidates.parquet \
  --output-dir reports/review/gpt-suggestions/input
```

Returned batches are structurally validated and classified with
`scripts/review/analyze_gpt_suggestions.py`. The UI discovers the resulting
`suggestions.parquet` automatically and offers changed text for one-click
acceptance only after listening to the audio.

An explicitly owner-approved suggestion pass can be converted to a complete
overlay with `scripts/review/apply_gpt_suggestions.py`. Existing native audio reviews
override text-only suggestions. See [review provenance](review-provenance.md);
never describe a
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

## Frozen training dataset

Dataset v3 supersedes v1 and v2 for new experiments. It contains 182,665 train,
1,000 validation, 999 test, 626 heldout-unused, and 3 excluded rows. Its
fingerprint is
`5cee7c7b91f5d7cab5ce10bab2ba85f6b18d49e1ab24fbeb50751d0fc374c31a`.
Train, validation, and test have zero speaker overlap.

V2 applied the complete owner-approved transcript overlay. V3 adds the
declared English correction in `configs/data/owner-text-overrides-v3.json` and
five owner edits saved after v2 was frozen. The reproducible sequence is:

```bash
PYTHONPATH=src python scripts/data/apply_text_overrides.py \
  --adjudications reports/review/gold-v1-owner-approved.jsonl \
  --overrides configs/data/owner-text-overrides-v3.json \
  --output reports/review/gold-v1-owner-approved-v3.jsonl

PYTHONPATH=src python scripts/data/finalize_dataset.py \
  --manifest data/versions/v1/manifest.parquet \
  --adjudications reports/review/gold-v1-owner-approved-v3.jsonl \
  --output-dir data/versions/v3
```

The full-corpus silence and clipping audit is documented in
[the audio audit](audio-audit.md).
Fifty risk-stratified crop proposals were reviewed as safe, but a matched local
training A/B showed worse early validation metrics and identical model FLOPs
with cropping. Consequently v3 retains immutable source audio and the baseline
training configuration leaves dynamic cropping disabled.

## Dataset v4 audio-verified evaluation set

Dataset v3 remains frozen. Because 295 validation/test references contain
text-only revisions that were not verified against audio, v4 restores the
conservatively normalized OpenSLR transcript as the neutral starting reference.
A self-contained native-listening queue contains all 295 disputed rows plus 100
deterministically selected unchanged controls:

```bash
PYTHONPATH=src python scripts/review/build_v4_review_queue.py \
  --manifest data/versions/v3/manifest.parquet \
  --output reports/review/v4-evaluation-queue.parquet \
  --controls 100

PYTHONPATH=src streamlit run scripts/review/review_app.py --server.port 8503 -- \
  --queue reports/review/v4-evaluation-queue.parquet \
  --output reports/review/v4-evaluation-adjudications.jsonl \
  --suggestions reports/review/v4-control-gpt/analysis/suggestions.parquet
```

In this queue, `Correct` means the displayed normalized OpenSLR transcript
matches the audio. For disputed rows the UI also shows the previous v3 revision;
use it only after listening confirms it. Otherwise edit the verified transcript.
Mark unusable audio as `Bad audio` and unresolved speech as `Uncertain`.

All 395 queued rows were reviewed by listening to their audio. Freeze v4 with:

```bash
PYTHONPATH=src python scripts/data/finalize_dataset_v4.py \
  --manifest data/versions/v3/manifest.parquet \
  --queue reports/review/v4-evaluation-queue.parquet \
  --adjudications reports/review/v4-evaluation-adjudications.jsonl \
  --output-dir data/versions/v4
```

The finalizer refuses partial or extra decisions. It resets reviewed validation
and test references to the normalized original, applies only audio-reviewed
edits, excludes reviewed bad/uncertain rows, and moves unheard evaluation rows
to `heldout_unreviewed`. Training rows, audio, and speaker assignments remain
unchanged.

The completed review produced 392 usable evaluation references: 206 validation
and 186 test. Three reviewed clips were excluded as bad audio. The remaining
1,604 v3 evaluation-speaker clips were not heard and are therefore retained as
`heldout_unreviewed`, not represented as gold evaluation data. The control
sample found transcript edits in 9 of 99 usable rows (9.09%), which is why
unheard rows were not kept in validation or test.

V4 leaves all 182,665 training rows and their audio/transcript content
unchanged. It also retains 626 `heldout_unused` rows and has six exclusions in
total. Train, validation, and test remain speaker-disjoint. Its fingerprint is
`d232747fbf019f06a6449404d3d0251e8f4547ed02471482c07d85014c81abdb`.

## Dataset v5: OpenSLR-only train/validation, external-only test (2026-09-07)

**v4 is unchanged and remains frozen.** Every completed experiment
(E000-E011) stays exactly reproducible against it. v5 is a separate,
additive restructuring, decided directly with the project owner in this
session: OpenSLR never appears in test again; test is built entirely from
two external, non-OpenSLR sources; and the train/validation boundary inside
OpenSLR was deliberately moved to land near an 80/10/10 split of the whole
corpus (OpenSLR + external test). Any training run built after this point
should use v5's manifests below, not v4's `train`/`validation` splits, to
get this design; anything using v4 directly still reproduces the historical
experiments exactly.

**Sources added:**

- **SPEAK-ASR YouTube corpus**: already indexed at
  `reports/dataset-audit/youtube-upstream/manifest.parquet` (4,037 rows,
  9.11h, 34 uploaders as a speaker/condition proxy, not verified individual
  voices). License was "unresolved, private-audit-only"; the owner approved
  private eval-only use on 2026-09-07 -- never train on it, never
  redistribute it.
- **Path Nirvana Sinhala TTS**: a separately cloned corpus, not part of this
  repo, at `/Users/tharupahan/Code/ASR-prior-works/sinhala-tts-dataset`
  (2 speakers -- `mettananda`, `oshadi` -- 6,386 clips, 13.61h, GPL-3, studio
  read speech, heavily Pali/Sanskrit-vocabulary religious texts, a different
  domain from OpenSLR's or YouTube's natural speech). Ingested with
  `scripts/data/prepare_pathnirvana_tts_manifest.py`, which resamples the
  source's native 22.05kHz mono audio to this project's 16kHz convention and
  runs it through the same `sinhala_asr.data.manifest.build_manifest_rows`
  pipeline used for every other source. Raw resampled copy:
  `data/raw/sinhala-tts-pathnirvana/data.parquet` (gitignored, 1.4GB).
  Manifest: `reports/dataset-audit/sinhala-tts-pathnirvana/manifest.parquet`
  (6,386 rows, 0 invalid).

**Build order** (each script additive; none modifies v4's manifest.parquet):

```bash
PYTHONPATH=.:src python scripts/data/prepare_pathnirvana_tts_manifest.py \
  --raw-output data/raw/sinhala-tts-pathnirvana/data.parquet \
  --manifest-output reports/dataset-audit/sinhala-tts-pathnirvana/manifest.parquet

PYTHONPATH=.:src python scripts/data/build_external_eval_manifest.py
# -> reports/dataset-audit/external-eval-v1/manifest.parquet (test, TTS+YouTube only)

PYTHONPATH=.:src python scripts/data/build_openslr_validation_v2.py
# -> reports/dataset-audit/openslr-validation-v2/manifest.parquet
#    (v4's validation+test+heldout_unreviewed+heldout_unused merged)

PYTHONPATH=.:src python scripts/data/build_train_val_8010.py
# -> reports/dataset-audit/openslr-train-v2/manifest.parquet (reduced train)
# -> reports/dataset-audit/openslr-validation-v2/manifest.parquet (overwritten,
#    enlarged again with 45 whole speakers moved out of train)
```

**Result:**

| Split | Source | Rows | Hours | Speakers | Share |
|---|---|---:|---:|---:|---:|
| Train | OpenSLR, `openslr-train-v2` | 165,055 | 199.49 | 426 | 80.7% |
| Validation | OpenSLR, `openslr-validation-v2` | 20,232 | 24.98 | 52 | 10.1% |
| Test | TTS + YouTube, `external-eval-v1` | 10,423 | 22.72 | 2 + 34 uploaders | 9.2% |

`dataset_split` in `external-eval-v1` is set to `test` (gated behind
`--unlock-test` in `scripts/evaluation/select_prediction_rows`, same
discipline as any other test split -- do not score iterative candidates
against it). `source_dataset` in `external-eval-v1` is relabeled per source
(`sinhala-tts-pathnirvana` / `youtube`) so
`sinhala_asr.evaluation.metrics.evaluate_rows`'s existing `by_source_dataset`
grouping reports the two test sources separately, alongside the aggregate,
with no new evaluation code.

**Known label-quality caveat, accepted deliberately, not silently:** of
validation's 20,232 rows, only 392 (the original v4 `validation`+`test`
rows) are audio-verified. The remaining 19,840 -- both the pre-existing
`heldout_unreviewed`/`heldout_unused` rows and the 45 newly moved train
speakers -- were never audio-reviewed. Their reference transcripts carry
whatever error rate the original v3->v4 audio-verification pass found
elsewhere in this corpus (9.09% of a spot-checked control sample had
transcript errors -- see the v4 section above). The owner explicitly chose
speed over review for this validation set ("just use them, no need to
review manually"). If a validation WER/CER number looks surprising, check
whether the disagreement lands in the unreviewed 19,840 before trusting it.

**A future training run selecting `openslr-train-v2`'s `train` split gets
199.49h across 426 speakers, not v4's 220.877h across 471** -- a real,
smaller pool than every completed experiment (E000-E011) trained on. This
is intentional per this section, not a regression to fix.
