#!/usr/bin/env python3
"""Export original labels and audio-verified targets for refiner benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sinhala_asr.review.store import load_adjudications


def build_benchmark_rows(
    queue: pd.DataFrame, adjudications: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    inputs = []
    truth = []
    for row in queue.to_dict("records"):
        sample_id = str(row["sample_id"])
        if not str(row.get("review_category", "")).startswith("disputed_"):
            continue
        record = adjudications.get(sample_id)
        if not record:
            raise ValueError(f"missing disputed-row decision: {sample_id}")
        decision = str(record["decision"])
        if decision in {"bad_audio", "uncertain"}:
            continue
        original = str(row["text_original"])
        verified = (
            str(record.get("text_corrected") or "").strip()
            if decision == "edited"
            else original
        )
        if not verified:
            raise ValueError(f"blank verified transcript: {sample_id}")
        inputs.append(
            {
                "sample_id": sample_id,
                "language_class": str(row["language_class"]),
                "original": original,
            }
        )
        truth.append(
            {
                "sample_id": sample_id,
                "original": original,
                "verified": verified,
                "should_change": verified != original,
                "review_decision": decision,
            }
        )
    return inputs, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--input-output", type=Path, required=True)
    parser.add_argument("--truth-output", type=Path, required=True)
    args = parser.parse_args()
    queue = pd.read_parquet(args.queue.expanduser().resolve())
    decisions = load_adjudications(args.adjudications.expanduser().resolve())
    inputs, truth = build_benchmark_rows(queue, decisions)
    for path, rows in ((args.input_output, inputs), (args.truth_output, truth)):
        output = path.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "usable_rows": len(truth),
                "verified_changes": sum(row["should_change"] for row in truth),
                "verified_unchanged": sum(not row["should_change"] for row in truth),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
