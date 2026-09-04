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


def render(summary: dict) -> str:
    strict = summary["strict"]
    canonical = summary["canonical"]
    intervals = summary["confidence_95"]
    return (
        "\n".join(
            [
                "# ASR Evaluation",
                "",
                f"- Rows: {strict['rows']}",
                f"- Strict WER: {strict['wer']:.2%} (95% CI {intervals['strict']['wer'][0]:.2%}–{intervals['strict']['wer'][1]:.2%})",
                f"- Strict CER: {strict['cer']:.2%} (95% CI {intervals['strict']['cer'][0]:.2%}–{intervals['strict']['cer'][1]:.2%})",
                f"- Canonical WER: {canonical['wer']:.2%} (95% CI {intervals['canonical']['wer'][0]:.2%}–{intervals['canonical']['wer'][1]:.2%})",
                f"- Canonical CER: {canonical['cer']:.2%} (95% CI {intervals['canonical']['cer'][0]:.2%}–{intervals['canonical']['cer'][1]:.2%})",
                "",
                "## Error labels",
                "",
                *[
                    f"- {key}: {value}"
                    for key, value in summary["error_label_counts"].items()
                ],
            ]
        )
        + "\n"
    )


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
