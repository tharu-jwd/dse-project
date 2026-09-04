#!/usr/bin/env python3
"""Validate returned GPT TSV batches and quantify their proposed changes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.evaluation.metrics import score_pair, strict_normalize
from sinhala_asr.review.suggestions import classify_change


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--returned-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = sorted(args.input_dir.expanduser().resolve().glob("batch-[0-9][0-9].tsv"))
    returned = sorted(
        args.returned_dir.expanduser().resolve().glob("batch-*-reviewed.tsv")
    )
    if len(inputs) != len(returned):
        raise SystemExit(
            f"batch count mismatch: {len(inputs)} input, {len(returned)} returned"
        )
    results = []
    seen: set[str] = set()
    for input_path, returned_path in zip(inputs, returned):
        original_rows = read_tsv(input_path)
        suggested_rows = read_tsv(returned_path)
        if len(original_rows) != len(suggested_rows):
            raise SystemExit(
                f"row count mismatch: {input_path.name} / {returned_path.name}"
            )
        for original, suggested in zip(original_rows, suggested_rows):
            sample_id = suggested.get("sample_id", "")
            if sample_id != original.get("sample_id"):
                raise SystemExit(
                    f"sample ID changed in {returned_path.name}: {sample_id}"
                )
            if sample_id in seen:
                raise SystemExit(f"duplicate sample ID: {sample_id}")
            seen.add(sample_id)
            if suggested.get("original_transcript") != original.get(
                "original_transcript"
            ):
                raise SystemExit(f"original transcript changed for {sample_id}")
            proposed = suggested.get("suggested_transcript", "").strip()
            if not proposed:
                raise SystemExit(f"blank suggestion for {sample_id}")
            source = original["original_transcript"]
            scores = score_pair(source, proposed, strict_normalize)
            results.append(
                {
                    "sample_id": sample_id,
                    "original_transcript": source,
                    "suggested_transcript": proposed,
                    "confidence": suggested.get("confidence", "").strip().lower(),
                    "reason": suggested.get("reason", "").strip(),
                    "change_class": classify_change(source, proposed),
                    **scores,
                }
            )
    classes = Counter(row["change_class"] for row in results)
    confidence = Counter(row["confidence"] for row in results)
    word_errors = sum(row["word_errors"] for row in results)
    word_units = sum(row["word_reference_units"] for row in results)
    character_errors = sum(row["character_errors"] for row in results)
    character_units = sum(row["character_reference_units"] for row in results)
    summary = {
        "rows": len(results),
        "changed_rows": len(results) - classes["unchanged"],
        "changed_fraction": (len(results) - classes["unchanged"]) / len(results),
        "change_classes": dict(sorted(classes.items())),
        "confidence": dict(sorted(confidence.items())),
        "original_to_suggestion_wer": word_errors / word_units,
        "original_to_suggestion_cer": character_errors / character_units,
        "word_edits": word_errors,
        "character_edits": character_errors,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(results),
        output_dir / "suggestions.parquet",
        compression="zstd",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# GPT suggestion analysis",
        "",
        f"- Rows: {summary['rows']}",
        f"- Changed: {summary['changed_rows']} ({summary['changed_fraction']:.2%})",
        f"- Original-to-suggestion WER: {summary['original_to_suggestion_wer']:.2%}",
        f"- Original-to-suggestion CER: {summary['original_to_suggestion_cer']:.2%}",
        "",
        "## Change classes",
        "",
    ]
    lines.extend(
        f"- {key}: {value}" for key, value in summary["change_classes"].items()
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
