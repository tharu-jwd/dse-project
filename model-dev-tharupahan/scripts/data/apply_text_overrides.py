#!/usr/bin/env python3
"""Apply explicit owner text corrections to an existing adjudication overlay."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sinhala_asr.review.store import load_adjudications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_adjudications(args.adjudications.expanduser().resolve())
    overrides = json.loads(
        args.overrides.expanduser().resolve().read_text(encoding="utf-8")
    )
    seen: set[str] = set()
    for override in overrides:
        sample_id = str(override["sample_id"])
        if sample_id in seen:
            raise SystemExit(f"duplicate override: {sample_id}")
        if sample_id not in records:
            raise SystemExit(f"override references unknown sample: {sample_id}")
        corrected = str(override["text_corrected"]).strip()
        if not corrected:
            raise SystemExit(f"blank corrected text: {sample_id}")
        seen.add(sample_id)
        records[sample_id] = dict(records[sample_id]) | {
            "decision": "edited",
            "text_corrected": corrected,
            "notes": str(override["reason"]),
            "review_method": "owner_approved_text_correction",
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(records):
            handle.write(json.dumps(records[sample_id], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(f"Applied {len(overrides)} owner text corrections to {len(records)} records")


if __name__ == "__main__":
    main()
