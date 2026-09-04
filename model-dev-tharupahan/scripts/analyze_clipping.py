#!/usr/bin/env python3
"""Measure clipping severity for manifest rows already flagged by peak amplitude."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from sinhala_asr.data.quality import clipping_severity


def analyze(task: tuple[str, str, float]) -> dict[str, Any]:
    sample_id, path, threshold = task
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    result = clipping_severity(samples, threshold=threshold)
    longest = int(result["longest_clipped_run_samples"])
    return {
        "sample_id": sample_id,
        "sample_rate": sample_rate,
        **result,
        "longest_clipped_run_ms": 1000 * longest / sample_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    source_tables: dict[str, list[dict[str, Any]]] = {}
    tasks = []
    for row in rows:
        if row.get("dataset_split") not in {"train", "validation", "test"}:
            continue
        if not row.get("is_clipped"):
            continue
        source_path = str(row["source_path"])
        if source_path not in source_tables:
            source_tables[source_path] = pq.read_table(
                source_path, columns=["audio"]
            ).to_pylist()
        audio = source_tables[source_path][int(row["source_row_index"])]["audio"]
        if not isinstance(audio, str):
            raise ValueError("clipping analysis currently requires path-backed audio")
        tasks.append((str(row["sample_id"]), audio, args.threshold))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze, tasks, chunksize=64))
    fractions = [float(row["clipped_fraction"]) for row in results]
    runs = [float(row["longest_clipped_run_ms"]) for row in results]
    summary = {
        "flagged_rows": len(results),
        "threshold": args.threshold,
        "rows_at_least_0_01pct_clipped": sum(value >= 0.0001 for value in fractions),
        "rows_at_least_0_1pct_clipped": sum(value >= 0.001 for value in fractions),
        "rows_at_least_1pct_clipped": sum(value >= 0.01 for value in fractions),
        "rows_with_run_at_least_1ms": sum(value >= 1 for value in runs),
        "rows_with_run_at_least_5ms": sum(value >= 5 for value in runs),
        "max_clipped_fraction": max(fractions, default=0.0),
        "max_clipped_run_ms": max(runs, default=0.0),
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(results),
        output / "measurements.parquet",
        compression="zstd",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
