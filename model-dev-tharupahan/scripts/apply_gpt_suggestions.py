#!/usr/bin/env python3
"""Create an owner-approved adjudication overlay from validated GPT suggestions."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from sinhala_asr.review.store import load_adjudications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suggestions", type=Path, required=True)
    parser.add_argument("--human-overrides", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = pq.read_table(args.manifest.expanduser().resolve()).to_pylist()
    candidate_ids = {
        str(row["sample_id"])
        for row in manifest
        if row.get("dataset_split") in {"validation_candidate", "test_candidate"}
    }
    suggestions = pq.read_table(args.suggestions.expanduser().resolve()).to_pylist()
    suggestion_ids = {str(row["sample_id"]) for row in suggestions}
    if suggestion_ids != candidate_ids:
        missing = candidate_ids - suggestion_ids
        unknown = suggestion_ids - candidate_ids
        raise SystemExit(
            f"suggestion ID mismatch: {len(missing)} missing, {len(unknown)} unknown"
        )
    reviewed_at = datetime.now(timezone.utc).isoformat()
    records = {}
    for row in suggestions:
        changed = row["suggested_transcript"] != row["original_transcript"]
        sample_id = str(row["sample_id"])
        records[sample_id] = {
            "sample_id": sample_id,
            "decision": "edited" if changed else "correct",
            "text_original": row["original_transcript"],
            "text_corrected": row["suggested_transcript"],
            "notes": (
                f"Owner-approved GPT-assisted text suggestion. {row['reason']}"
                if changed
                else "Owner-approved GPT-assisted text review; unchanged."
            ),
            "review_method": "owner_approved_gpt_assisted_text_only",
            "suggestion_confidence": row["confidence"],
            "suggestion_change_class": row["change_class"],
            "reviewed_at_utc": reviewed_at,
        }
    overrides = {}
    if args.human_overrides:
        overrides = load_adjudications(args.human_overrides.expanduser().resolve())
        unknown = set(overrides) - candidate_ids
        if unknown:
            raise SystemExit(
                f"human overrides contain {len(unknown)} unknown sample IDs"
            )
        for sample_id, record in overrides.items():
            records[sample_id] = dict(record) | {"review_method": "native_audio_review"}
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(records):
            handle.write(json.dumps(records[sample_id], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(
        f"Wrote {len(records)} adjudications with {len(overrides)} native-audio overrides"
    )
    print(output)


if __name__ == "__main__":
    main()
