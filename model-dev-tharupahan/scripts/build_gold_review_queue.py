#!/usr/bin/env python3
"""Build the self-contained native-review queue for validation/test candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from sinhala_asr.review.queue import attach_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    candidates = [
        row
        for row in rows
        if row.get("dataset_split") in {"validation_candidate", "test_candidate"}
    ]
    if not candidates:
        raise SystemExit("manifest contains no validation/test candidates")
    attached = attach_audio(candidates)
    for row in attached:
        row["review_category"] = row["dataset_split"]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(attached)
    frame.to_parquet(output, compression="zstd", index=False)
    print(frame.groupby("review_category").size().to_string())
    print(f"Queue: {output}")


if __name__ == "__main__":
    main()
