#!/usr/bin/env python3
"""Repackage the full-fine-tune search's OpenSLR rows with embedded audio.

v5's openslr-train-v2/openslr-validation-v2 manifests store audio by
reference (source_path -> data/indexes/openslr52.parquet, itself a path-only
index into individual local FLAC files under data/raw/openslr52/) -- fine
locally, but not portable to Kaggle. This resolves real audio bytes for a
small, deterministic slice (the 1,440-row Sinhala pilot pool plus a 200-row
validation slice -- the same rows build_full_finetune_search_manifests.py
already selected) and writes self-contained raw parquets + manifests through
the normal sinhala_asr.data.manifest pipeline, so the result uploads to
Kaggle as ordinary standalone files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.data.manifest import _audio_bytes, build_manifest_rows, write_manifest

SINHALA_TRAIN = "reports/dataset-audit/openslr-train-v2/manifest.parquet"
VALIDATION = "reports/dataset-audit/openslr-validation-v2/manifest.parquet"

SINHALA_POOL_ROWS = 1440  # matches build_full_finetune_search_manifests.py's largest ratio
VALIDATION_ROWS = 200

SINHALA_RAW_OUTPUT = "data/raw/full-finetune-search-sinhala-pilot/data.parquet"
SINHALA_MANIFEST_OUTPUT = "reports/dataset-audit/full-finetune-search-sinhala-pilot/manifest.parquet"
VALIDATION_RAW_OUTPUT = "data/raw/full-finetune-search-validation-pilot/data.parquet"
VALIDATION_MANIFEST_OUTPUT = "reports/dataset-audit/full-finetune-search-validation-pilot/manifest.parquet"


def _resolve_audio(row: dict) -> bytes:
    source_path = Path(row["source_path"])
    raw_row = pq.read_table(source_path).slice(int(row["source_row_index"]), 1).to_pylist()[0]
    return _audio_bytes(raw_row.get("audio"), source_path.parent)


def _package(manifest_path: str, n_rows: int, raw_output: str, manifest_output: str) -> None:
    df = pd.read_parquet(manifest_path).sort_values("sample_id").head(n_rows)
    records = []
    for row in df.to_dict("records"):
        audio_bytes = _resolve_audio(row)
        records.append(
            {
                "audio": {"bytes": audio_bytes},
                "text": row["text_canonical"],
                "speaker_id": row["speaker_id"],
                "source_dataset": "openslr52",
                "source_record_id": row["sample_id"],
                "domain": "openslr52_v5",
            }
        )
    Path(raw_output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), raw_output, compression="zstd")
    rows = build_manifest_rows(split="unsplit", path=Path(raw_output))
    write_manifest(rows, Path(manifest_output))
    invalid = sum(1 for r in rows if not r["is_valid"])
    print(f"{manifest_output}: {len(rows)} rows, {invalid} invalid")


def main() -> None:
    _package(SINHALA_TRAIN, SINHALA_POOL_ROWS, SINHALA_RAW_OUTPUT, SINHALA_MANIFEST_OUTPUT)
    _package(VALIDATION, VALIDATION_ROWS, VALIDATION_RAW_OUTPUT, VALIDATION_MANIFEST_OUTPUT)


if __name__ == "__main__":
    main()
