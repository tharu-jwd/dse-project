#!/usr/bin/env python3
"""Export deterministic training-only transcript batches for label refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sinhala", type=int, default=10)
    parser.add_argument("--english", type=int, default=10)
    args = parser.parse_args()
    rows = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    train = [row for row in rows if row.get("dataset_split") == "train"]
    selected = []
    for language_class, quota in (
        ("sinhala_only", args.sinhala),
        ("latin_only", args.english),
    ):
        candidates = sorted(
            (row for row in train if row.get("language_class") == language_class),
            key=lambda row: str(row["sample_id"]),
        )
        if len(candidates) < quota:
            raise SystemExit(f"only {len(candidates)} {language_class} training rows")
        selected.extend(candidates[:quota])
    output_rows = [
        {
            "sample_id": str(row["sample_id"]),
            "language_class": str(row["language_class"]),
            "original": str(row["text_canonical"]),
        }
        for row in selected
    ]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(output_rows)} training-only rows to {output}")


if __name__ == "__main__":
    main()
