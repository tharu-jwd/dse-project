#!/usr/bin/env python3
"""Create non-destructive crop proposals from boundary-silence measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--leading-margin", type=float, default=0.25)
    parser.add_argument("--trailing-margin", type=float, default=0.40)
    args = parser.parse_args()
    manifest = {
        str(row["sample_id"]): row
        for row in pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
        if row.get("dataset_split") in {"train", "validation", "test"}
    }
    measurements = pq.read_table(args.measurements.expanduser().resolve()).to_pylist()
    proposals = []
    original_seconds = 0.0
    proposed_seconds = 0.0
    for measured in measurements:
        sample_id = str(measured["sample_id"])
        row = manifest[sample_id]
        duration = float(row["duration_seconds"])
        start = max(
            0.0, float(measured["leading_silence_seconds"]) - args.leading_margin
        )
        end = min(
            duration,
            duration
            - max(
                0.0, float(measured["trailing_silence_seconds"]) - args.trailing_margin
            ),
        )
        if end <= start:
            start, end = 0.0, duration
        original_seconds += duration
        proposed_seconds += end - start
        proposals.append(
            {
                "sample_id": sample_id,
                "crop_start_seconds": start,
                "crop_end_seconds": end,
                "original_duration_seconds": duration,
                "proposed_duration_seconds": end - start,
                "saved_seconds": duration - (end - start),
            }
        )
    if set(manifest) != {str(row["sample_id"]) for row in measurements}:
        raise SystemExit(
            "manifest and silence measurements do not have identical sample coverage"
        )
    summary = {
        "rows": len(proposals),
        "leading_margin_seconds": args.leading_margin,
        "trailing_margin_seconds": args.trailing_margin,
        "original_hours": original_seconds / 3600,
        "proposed_hours": proposed_seconds / 3600,
        "saved_hours": (original_seconds - proposed_seconds) / 3600,
        "saved_fraction": (original_seconds - proposed_seconds) / original_seconds,
        "source_audio_modified": False,
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(proposals),
        output / "proposals.parquet",
        compression="zstd",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
