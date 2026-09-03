#!/usr/bin/env python3
"""Apply a complete native-review overlay and lock validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.review.finalize import finalize_rows
from sinhala_asr.review.store import load_adjudications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    records = load_adjudications(args.adjudications.expanduser().resolve())
    finalized, summary = finalize_rows(rows, records)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty dataset version: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(finalized),
        output_dir / "manifest.parquet",
        compression="zstd",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
