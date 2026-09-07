#!/usr/bin/env python3
"""Build a manifest for the 1,111 English teacher-replay clips.

Merges the local English-replay audio shards
(reports/kaggle/e003-input-dataset/english-replay-part-*.parquet, hash-verified
against ENGLISH_SHA256/scripts/training/run_e007_kaggle.py's own convention)
with the local teacher-labels parquet
(reports/experiments/e004-teacher-behavior-replay/teacher-labels/
e004-english-teacher-labels.parquet, hash-verified against TEACHER_SHA256),
replacing each clip's raw ground-truth `text` with the untouched model's own
`teacher_text` -- this is what makes it "teacher-behavior replay" rather than
plain English fine-tuning (see E004's report). Same mechanism E004-E007 used,
rebuilt here as a standalone, reusable manifest instead of one-off Kaggle
job code, so new experiments (e.g. the full-fine-tune LR/replay-ratio search)
can mix it into their own training manifests at any ratio.
"""

from __future__ import annotations

import glob
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.data.manifest import build_manifest_rows, write_manifest

ENGLISH_SHARDS = "reports/kaggle/e003-input-dataset/english-replay-part-*.parquet"
TEACHER_LABELS = (
    "reports/experiments/e004-teacher-behavior-replay/teacher-labels/"
    "e004-english-teacher-labels.parquet"
)
TEACHER_SHA256 = "5f45c712867f430ab2e911cfa8950634c69b5897689b082486c8c6f1fc45c0f6"
RAW_OUTPUT = "data/raw/teacher-replay-english/data.parquet"
MANIFEST_OUTPUT = "reports/dataset-audit/teacher-replay-english/manifest.parquet"


def _digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    teacher_hash = _digest(TEACHER_LABELS)
    if teacher_hash != TEACHER_SHA256:
        raise SystemExit(f"teacher-labels hash mismatch: {teacher_hash} != {TEACHER_SHA256}")
    teachers = pd.read_parquet(TEACHER_LABELS).set_index("sample_id")

    shards = [pd.read_parquet(p) for p in sorted(glob.glob(ENGLISH_SHARDS))]
    english = pd.concat(shards, ignore_index=True)
    if len(english) != 1111 or english["sample_id"].nunique() != 1111:
        raise SystemExit(f"expected 1,111 unique English rows, got {len(english)}")

    replaced = []
    for row in english.to_dict("records"):
        teacher = teachers.loc[row["sample_id"]]
        if teacher["audio_sha256"] != row["audio_sha256"] or teacher["text"] != row["text"]:
            raise SystemExit(f"teacher/source mismatch: {row['sample_id']}")
        replaced.append(
            {
                "audio": {"bytes": row["audio"]},
                "text": teacher["teacher_text"],
                "speaker_id": row["speaker_id"],
                "source_dataset": "librispeech_teacher_replay",
                "source_record_id": row["sample_id"],
                "domain": "english_teacher_replay",
            }
        )

    Path(RAW_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(replaced), RAW_OUTPUT, compression="zstd")
    print(f"wrote {RAW_OUTPUT}: {len(replaced)} rows")

    rows = build_manifest_rows(split="unsplit", path=Path(RAW_OUTPUT))
    write_manifest(rows, Path(MANIFEST_OUTPUT))
    invalid = sum(1 for r in rows if not r["is_valid"])
    print(f"wrote {MANIFEST_OUTPUT}: {len(rows)} rows, {invalid} invalid")


if __name__ == "__main__":
    main()
