#!/usr/bin/env python3
"""Combine audit manifests and report exact overlap between source datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from sinhala_asr.data.combine import (
    combine_manifests,
    cross_source_report,
    write_cross_source_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    table = combine_manifests(
        [path.expanduser().resolve() for path in args.manifest],
        output_dir / "manifest.parquet",
    )
    report = cross_source_report(table)
    write_cross_source_report(report, output_dir)
    print((output_dir / "overlap.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
