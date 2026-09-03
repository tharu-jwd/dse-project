#!/usr/bin/env python3
"""Measure leading/trailing silence without modifying source audio."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from sinhala_asr.data.silence import boundary_silence


def analyze(task: tuple[str, str, float, float]) -> dict[str, Any]:
    sample_id, path, threshold, frame_ms = task
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return {
        "sample_id": sample_id,
        **boundary_silence(
            samples, sample_rate, threshold=threshold, frame_ms=frame_ms
        ),
    }


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)] if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append")
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="Linear RMS; 0.01 is -40 dBFS"
    )
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    if args.split:
        manifest = [
            row for row in manifest if row.get("dataset_split") in set(args.split)
        ]
    source_tables: dict[str, list[dict[str, Any]]] = {}
    tasks = []
    for row in manifest:
        source_path = str(row["source_path"])
        if source_path not in source_tables:
            source_tables[source_path] = pq.read_table(
                source_path, columns=["audio"]
            ).to_pylist()
        audio = source_tables[source_path][int(row["source_row_index"])]["audio"]
        if not isinstance(audio, str):
            raise ValueError("boundary analysis currently requires path-backed audio")
        tasks.append((str(row["sample_id"]), audio, args.threshold, args.frame_ms))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze, tasks, chunksize=128))
    leading = [float(row["leading_silence_seconds"]) for row in results]
    trailing = [float(row["trailing_silence_seconds"]) for row in results]
    fractions = [float(row["boundary_silence_fraction"]) for row in results]
    summary = {
        "rows": len(results),
        "threshold_linear_rms": args.threshold,
        "threshold_dbfs": 20 * __import__("math").log10(args.threshold),
        "frame_ms": args.frame_ms,
        "leading_seconds": {
            "mean": mean(leading),
            "p50": percentile(leading, 0.5),
            "p95": percentile(leading, 0.95),
            "p99": percentile(leading, 0.99),
            "max": max(leading),
        },
        "trailing_seconds": {
            "mean": mean(trailing),
            "p50": percentile(trailing, 0.5),
            "p95": percentile(trailing, 0.95),
            "p99": percentile(trailing, 0.99),
            "max": max(trailing),
        },
        "clips_over_1s_leading": sum(value > 1 for value in leading),
        "clips_over_1s_trailing": sum(value > 1 for value in trailing),
        "clips_over_40pct_boundary_silence": sum(value > 0.4 for value in fractions),
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(results),
        output_dir / "measurements.parquet",
        compression="zstd",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
