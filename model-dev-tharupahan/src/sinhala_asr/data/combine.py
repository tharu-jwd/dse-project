"""Combine audited manifests and measure overlap between source datasets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def combine_manifests(paths: list[Path], output_path: Path) -> pa.Table:
    """Concatenate manifests without changing their rows or source metadata."""
    if not paths:
        raise ValueError("at least one manifest is required")
    tables = [pq.read_table(path) for path in paths]
    table = pa.concat_tables(tables, promote_options="default")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")
    return table


def _pairwise_overlap(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row.get(field)
        source = row.get("source_dataset")
        if value and source:
            values[str(source)].add(str(value))
    results = []
    for left, right in combinations(sorted(values), 2):
        overlap = values[left] & values[right]
        results.append(
            {
                "left": left,
                "right": right,
                "count": len(overlap),
                "left_fraction": len(overlap) / len(values[left])
                if values[left]
                else 0.0,
                "right_fraction": len(overlap) / len(values[right])
                if values[right]
                else 0.0,
                "example_values": sorted(overlap)[:10],
            }
        )
    return results


def cross_source_report(table: pa.Table) -> dict[str, Any]:
    rows = table.to_pylist()
    canonical = "\n".join(
        sorted(
            f"{row.get('source_dataset')}\t{row.get('sample_id')}\t"
            f"{row.get('audio_pcm_sha256')}\t{row.get('transcript_sha256')}"
            for row in rows
        )
    )
    return {
        "combined_fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "total_rows": len(rows),
        "source_rows": {
            source: sum(row.get("source_dataset") == source for row in rows)
            for source in sorted({str(row.get("source_dataset")) for row in rows})
        },
        "decoded_audio_overlap": _pairwise_overlap(rows, "audio_pcm_sha256"),
        "encoded_audio_overlap": _pairwise_overlap(rows, "audio_sha256"),
        "transcript_overlap": _pairwise_overlap(rows, "transcript_sha256"),
        "sample_id_overlap": _pairwise_overlap(rows, "sample_id"),
    }


def write_cross_source_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlap.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Cross-source overlap",
        "",
        f"- Rows: {report['total_rows']}",
        f"- Fingerprint: `{report['combined_fingerprint']}`",
        "",
        "## Exact decoded-audio overlap",
        "",
        "| Sources | Shared hashes | Left | Right |",
        "|---|---:|---:|---:|",
    ]
    for item in report["decoded_audio_overlap"]:
        lines.append(
            f"| {item['left']} / {item['right']} | {item['count']} | "
            f"{item['left_fraction']:.2%} | {item['right_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Exact normalized-transcript overlap",
            "",
            "| Sources | Shared hashes | Left | Right |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in report["transcript_overlap"]:
        lines.append(
            f"| {item['left']} / {item['right']} | {item['count']} | "
            f"{item['left_fraction']:.2%} | {item['right_fraction']:.2%} |"
        )
    (output_dir / "overlap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
