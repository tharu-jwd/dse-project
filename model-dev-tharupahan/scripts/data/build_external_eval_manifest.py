#!/usr/bin/env python3
"""Combine the three external/held-out eval sources into one manifest.

Sources, per docs/project/plan.md item 2 and item 7:
1. sinhala-tts-pathnirvana -- Path Nirvana studio TTS corpus (2 speakers,
   13.61h), read speech, GPL-3, eval-only (never train/redistribute).
2. youtube -- SPEAK-ASR YouTube corpus (34 uploaders, 9.1h), license
   unresolved, approved for private eval-only use.
3. openslr52_seen_speaker_holdout -- one held-back utterance per v4 training
   speaker (471 speakers, 471 rows, ~0.57h), a same-speaker/unseen-utterance
   diagnostic. NOTE: only valid as a genuine holdout for a future training
   run that explicitly excludes these sample_ids; every completed experiment
   (E000-E011) already trained on these rows and cannot be scored against
   this source as a real held-out check -- their numbers on this slice would
   be in-sample, not generalization evidence.

`eval_source` is the field to group/report by; `source_dataset` is preserved
from each source's own manifest for provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SOURCES = {
    "sinhala-tts-pathnirvana": "reports/dataset-audit/sinhala-tts-pathnirvana/manifest.parquet",
    "youtube": "reports/dataset-audit/youtube-upstream/manifest.parquet",
}
# openslr52_seen_speaker_holdout deliberately excluded from the test bundle per
# user direction: OpenSLR is train+validation only, never test. The holdout
# manifest still exists (reports/dataset-audit/openslr-seen-speaker-holdout/)
# in case a same-speaker diagnostic is wanted later, but it is not part of
# the test set.
OUTPUT = "reports/dataset-audit/external-eval-v1/manifest.parquet"


def main() -> None:
    frames = []
    for eval_source, path in SOURCES.items():
        df = pd.read_parquet(path)
        df = df.drop(columns=[c for c in ("dataset_split", "exclusion_reason") if c in df.columns])
        # Relabel source_dataset to the distinguishing eval_source name so the
        # existing evaluate_rows()/aggregate() by_source_dataset grouping
        # (src/sinhala_asr/evaluation/metrics.py) separates these three sets
        # with zero new evaluation code -- it already groups by source_dataset.
        df["source_dataset_original"] = df["source_dataset"]
        df["source_dataset"] = eval_source
        # Per user direction: TTS + YouTube ARE the test set now (OpenSLR is
        # never used for testing). This inherits the same protective gate as
        # any other "test" split in select_prediction_rows() -- it requires
        # --unlock-test, so it isn't scored repeatedly during iteration,
        # same discipline as the original frozen OpenSLR test set.
        df["dataset_split"] = "test"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    dupes = combined["sample_id"].duplicated().sum()
    if dupes:
        raise SystemExit(f"refusing to write: {dupes} duplicate sample_ids across sources")

    fingerprint = hashlib.sha256(
        "\n".join(sorted(combined["sample_id"].astype(str))).encode("utf-8")
    ).hexdigest()

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(combined, preserve_index=False), OUTPUT, compression="zstd")

    print(f"wrote {OUTPUT}: {len(combined)} rows")
    print("fingerprint:", fingerprint)
    for eval_source in SOURCES:
        sub = combined[combined["source_dataset"] == eval_source]
        print(
            f"  {eval_source}: {len(sub)} rows, "
            f"{sub['duration_seconds'].sum() / 3600:.2f}h, "
            f"{sub['speaker_id'].nunique() if sub['speaker_id'].notna().any() else 0} speaker_ids, "
            f"{sub['uploader'].nunique() if sub['uploader'].notna().any() else 0} uploaders"
        )
    print("TOTAL:", f"{combined['duration_seconds'].sum() / 3600:.2f}h")


if __name__ == "__main__":
    main()
