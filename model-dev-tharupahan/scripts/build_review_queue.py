#!/usr/bin/env python3
"""Build a deterministic, self-contained audio queue for native-speaker review."""

from __future__ import annotations

import argparse
from pathlib import Path

from sinhala_asr.review.queue import build_review_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota", type=int, default=100, help="Maximum unique samples per category")
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    frame = build_review_queue(
        args.manifest.expanduser().resolve(),
        args.output.expanduser().resolve(),
        quota=args.quota,
        seed=args.seed,
    )
    print(frame.groupby("review_category").size().to_string())
    print(f"Queue: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
