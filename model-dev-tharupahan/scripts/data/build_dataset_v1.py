#!/usr/bin/env python3
"""Build licensed OpenSLR training data and speaker-disjoint gold-set candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.data.splits import make_speaker_disjoint_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    rows = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    sources = sorted({str(row["source_dataset"]) for row in rows})
    if sources != ["openslr52"]:
        raise SystemExit(
            f"v1 accepts only the licensed openslr52 manifest, got: {sources}"
        )
    assigned, summary = make_speaker_disjoint_candidates(
        rows, candidate_rows=args.candidate_rows, seed=args.seed
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(assigned),
        output_dir / "manifest.parquet",
        compression="zstd",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["split_counts"], indent=2))
    print(f"Split fingerprint: {summary['split_fingerprint']}")


if __name__ == "__main__":
    main()
