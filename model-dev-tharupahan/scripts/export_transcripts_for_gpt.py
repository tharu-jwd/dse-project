#!/usr/bin/env python3
"""Export review transcripts as aligned UTF-8 TSV batches for GPT suggestions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


FIELDS = (
    "sample_id",
    "original_transcript",
    "suggested_transcript",
    "confidence",
    "reason",
)


def write_tsv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    queue = pd.read_parquet(args.queue.expanduser().resolve())
    rows = [
        {
            "sample_id": str(row.sample_id),
            "original_transcript": str(row.text_original),
            "suggested_transcript": "",
            "confidence": "",
            "reason": "",
        }
        for row in queue.itertuples(index=False)
    ]
    output_dir = args.output_dir.expanduser().resolve()
    write_tsv(rows, output_dir / "all-transcripts.tsv")
    for start in range(0, len(rows), args.batch_size):
        number = start // args.batch_size + 1
        write_tsv(
            rows[start : start + args.batch_size],
            output_dir / f"batch-{number:02d}.tsv",
        )
    prompt = """Review the uploaded UTF-8 TSV as Sinhala transcript text only.

For every row, preserve sample_id and row order exactly. Copy original_transcript into suggested_transcript, changing it only to fix a clear spelling, Unicode, spacing, or punctuation error. Preserve meaning, wording, colloquial speech, English words, names, and digits. Do not translate, formalize, paraphrase, add unheard context, or delete a word merely because the sentence seems unusual. Set confidence to high, medium, or low. Briefly describe an actual change in reason; leave reason blank when unchanged. Return a downloadable UTF-8 TSV with exactly the same columns and number of rows. Do not return a Markdown table.

You cannot hear the audio, so these are suggestions rather than ground truth.
"""
    (output_dir / "PROMPT.txt").write_text(prompt, encoding="utf-8")
    print(
        f"Exported {len(rows)} rows in {(len(rows) + args.batch_size - 1) // args.batch_size} batches"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
