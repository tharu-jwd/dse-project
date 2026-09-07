#!/usr/bin/env python3
"""Carve a seen-speaker holdout out of v4's `train` split, additively.

This does NOT modify data/versions/v4/manifest.parquet. v4 stays frozen and
every completed experiment (E000-E011) remains exactly reproducible against
it. This script only *records*, in a separate file, one held-back utterance
per training speaker (chosen deterministically by lowest sample_id) that a
FUTURE training run may choose to exclude from its own training row
selection, so that later checkpoints can be scored on speakers they were
trained on but sentences they never saw -- separating "bad at Sinhala" from
"bad at unseen voices" (see docs/project/plan.md, item 2 and item 7).

Existing experiments (E000-E011) are unaffected: they already trained on
100% of v4's train split, including these rows, so this holdout is not a
valid held-out check for any of them -- only for training runs that are
built to actually exclude it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

V4_MANIFEST = Path("data/versions/v4/manifest.parquet")
OUTPUT_MANIFEST = Path("reports/dataset-audit/openslr-seen-speaker-holdout/manifest.parquet")
OUTPUT_SUMMARY = Path("reports/dataset-audit/openslr-seen-speaker-holdout/summary.json")


def main() -> None:
    df = pd.read_parquet(V4_MANIFEST)
    train = df[df["dataset_split"] == "train"].copy()
    train = train.sort_values("sample_id")
    holdout = train.groupby("speaker_id", as_index=False, group_keys=False).head(1)
    holdout = holdout.sort_values("sample_id").reset_index(drop=True)

    fingerprint = hashlib.sha256(
        "\n".join(sorted(holdout["sample_id"].astype(str))).encode("utf-8")
    ).hexdigest()

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(holdout, preserve_index=False), OUTPUT_MANIFEST, compression="zstd")

    summary = {
        "source": "data/versions/v4/manifest.parquet (dataset_split == 'train')",
        "selection_method": "one row per speaker_id, lowest sample_id, deterministic",
        "row_count": int(len(holdout)),
        "speaker_count": int(holdout["speaker_id"].nunique()),
        "total_hours": float(holdout["duration_seconds"].sum() / 3600),
        "sample_id_set_fingerprint_sha256": fingerprint,
        "note": (
            "Additive only -- v4's own train/validation/test/heldout splits are "
            "unchanged. Existing experiments (E000-E011) already trained on these "
            "rows and cannot be scored against this holdout as a valid seen-speaker "
            "check. Only a training run that explicitly excludes these sample_ids "
            "from its own training selection can be scored against this set as a "
            "genuine same-speaker/held-back-utterance diagnostic."
        ),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
