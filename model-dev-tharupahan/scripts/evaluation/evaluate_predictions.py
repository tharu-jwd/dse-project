#!/usr/bin/env python3
"""Evaluate prediction Parquet/JSONL with strict, canonical, and subgroup metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.parquet as pq

from sinhala_asr.evaluation.metrics import evaluate_rows


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix in {".jsonl", ".ndjson"}:
        return pajson.read_json(path).to_pylist()
    raise ValueError("predictions must be .parquet or .jsonl")


def _rate_line(metrics: dict, unit: str) -> str:
    total = metrics.get(f"{unit}_reference_units") or 0
    if not total:
        return f"- {unit.capitalize()} substitution/deletion/insertion rate: n/a (no reference units)"
    sub = metrics[f"{unit}_substitutions"] / total
    dele = metrics[f"{unit}_deletions"] / total
    ins = metrics[f"{unit}_insertions"] / total
    return (
        f"- {unit.capitalize()} substitution/deletion/insertion rate: "
        f"{sub:.2%} / {dele:.2%} / {ins:.2%}"
    )


def _subgroup_table(summary: dict, field: str, title: str) -> list[str]:
    key = f"by_{field}"
    if key not in summary:
        return []
    lines = [f"## {title}", "", "| Group | Rows | Canonical WER | Canonical CER |", "|---|---:|---:|---:|"]
    for group, metrics in summary[key].items():
        canonical = metrics["canonical"]
        lines.append(
            f"| {group} | {canonical['rows']} | {canonical['wer']:.2%} | {canonical['cer']:.2%} |"
        )
    lines.append("")
    return lines


def render(summary: dict) -> str:
    strict = summary["strict"]
    canonical = summary["canonical"]
    intervals = summary["confidence_95"]
    lines = [
        "# ASR Evaluation",
        "",
        f"- Rows: {strict['rows']}",
        f"- Strict WER: {strict['wer']:.2%} (95% CI {intervals['strict']['wer'][0]:.2%}–{intervals['strict']['wer'][1]:.2%})",
        f"- Strict CER: {strict['cer']:.2%} (95% CI {intervals['strict']['cer'][0]:.2%}–{intervals['strict']['cer'][1]:.2%})",
        f"- Canonical WER: {canonical['wer']:.2%} (95% CI {intervals['canonical']['wer'][0]:.2%}–{intervals['canonical']['wer'][1]:.2%})",
        f"- Canonical CER: {canonical['cer']:.2%} (95% CI {intervals['canonical']['cer'][0]:.2%}–{intervals['canonical']['cer'][1]:.2%})",
        _rate_line(canonical, "word"),
        _rate_line(canonical, "character"),
        "",
        "## Error labels",
        "",
        *[
            f"- {key}: {value}"
            for key, value in summary["error_label_counts"].items()
        ],
        "",
    ]
    lines += _subgroup_table(summary, "speaker_id", "By speaker")
    lines += _subgroup_table(summary, "duration_bucket", "By audio duration (data-driven quartiles)")
    lines += _subgroup_table(summary, "transcript_length_bucket", "By reference length (data-driven quartiles)")
    lines += _subgroup_table(summary, "source_dataset", "By source dataset")
    lines += _subgroup_table(summary, "language_class", "By language class")
    lines += _subgroup_table(summary, "dataset_split", "By dataset split")
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    path = args.predictions.expanduser().resolve()
    rows = read_rows(path)
    required = {"sample_id", "reference", "prediction"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    scored, summary = evaluate_rows(
        rows, bootstrap_iterations=args.bootstrap_iterations
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(scored), output_dir / "scored.parquet", compression="zstd"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render(summary)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
