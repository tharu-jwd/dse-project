#!/usr/bin/env python3
"""Merge v4's validation+test rows into one enlarged validation-only set.

Per user direction: OpenSLR is train+validation only, never test (TTS +
YouTube are the test set now -- see build_external_eval_manifest.py). v4's
old validation (206 rows, 3 speakers) and test (186 rows, 4 speakers) splits
were both audio-verified and both drawn from the same disjoint 7-speaker
holdout pool, so merging them is safe: still fully disjoint from the 471
train speakers, still fully audio-verified, just a bigger validation set
(392 rows / 7 speakers instead of 206 / 3).

Also includes heldout_unreviewed (1,604 rows) and heldout_unused (626 rows)
by explicit user direction ("just use them, no need to review manually").
Flagged here for the record, not as a blocker: those 1,604 rows are NOT
confirmed audio-reviewed, so their reference transcripts carry a real,
un-quantified error rate -- this project's dataset v4 was originally built
specifically because unreviewed transcripts were unreliable. Any validation
WER/CER computed on this enlarged set is therefore noisier than the
392-row audio-verified core; if a number from this set looks surprising,
check whether it lands in the unreviewed 1,604 before trusting it.

v4's own manifest.parquet is NOT modified -- this is purely additive.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

V4_MANIFEST = "data/versions/v4/manifest.parquet"
OUTPUT = "reports/dataset-audit/openslr-validation-v2/manifest.parquet"


def main() -> None:
    df = pd.read_parquet(V4_MANIFEST)
    merged = df[
        df["dataset_split"].isin(
            ["validation", "test", "heldout_unreviewed", "heldout_unused"]
        )
    ].copy()
    merged["is_audio_reviewed"] = merged["dataset_split"].isin(["validation", "test"])
    merged["dataset_split"] = "validation"
    merged = merged.sort_values("sample_id").reset_index(drop=True)

    fingerprint = hashlib.sha256(
        "\n".join(sorted(merged["sample_id"].astype(str))).encode("utf-8")
    ).hexdigest()

    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), OUTPUT, compression="zstd")
    print(f"wrote {OUTPUT}: {len(merged)} rows")
    print(f"hours: {merged['duration_seconds'].sum() / 3600:.3f}")
    print(f"speakers: {sorted(merged['speaker_id'].unique())}")
    reviewed = merged[merged["is_audio_reviewed"]]
    unreviewed = merged[~merged["is_audio_reviewed"]]
    print(
        f"audio-reviewed: {len(reviewed)} rows / {reviewed['duration_seconds'].sum() / 3600:.3f}h  |  "
        f"unreviewed: {len(unreviewed)} rows / {unreviewed['duration_seconds'].sum() / 3600:.3f}h"
    )
    print("fingerprint:", fingerprint)


if __name__ == "__main__":
    main()
