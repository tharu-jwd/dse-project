#!/usr/bin/env python3
"""Build the self-contained v4 audio review queue from frozen dataset v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from sinhala_asr.review.queue import attach_audio
from sinhala_asr.review.v4 import select_v4_review_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.controls < 0:
        raise SystemExit("controls must be non-negative")
    rows = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    selected = attach_audio(
        select_v4_review_rows(rows, control_rows=args.controls, seed=args.seed)
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(selected)
    frame.to_parquet(output, compression="zstd", index=False)
    print(frame.groupby("review_category").size().to_string())
    print(f"Total: {len(frame)}")
    print(f"Queue: {output}")


if __name__ == "__main__":
    main()
