#!/usr/bin/env python3
"""Create dataset v4 from complete audio review of disputed v3 references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.review.store import load_adjudications, validate_queue
from sinhala_asr.review.v4 import finalize_v4_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    queue = pd.read_parquet(args.queue.expanduser().resolve())
    validate_queue(queue)
    selected_ids = set(queue["sample_id"].astype(str))
    decisions = load_adjudications(args.adjudications.expanduser().resolve())
    finalized, summary = finalize_v4_rows(manifest, selected_ids, decisions)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty dataset version: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(finalized), output / "manifest.parquet", compression="zstd")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
