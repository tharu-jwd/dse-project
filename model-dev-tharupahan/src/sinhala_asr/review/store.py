"""Versioned, resumable adjudication storage."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DECISIONS = ("correct", "edited", "bad_audio", "mismatch", "duplicate", "uncertain")
# Legacy decisions remain valid so previously saved overlays stay readable.
REVIEW_DECISIONS = ("correct", "edited", "bad_audio", "uncertain")
REQUIRED_QUEUE_COLUMNS = {"sample_id", "text_original"}


def validate_queue(frame: pd.DataFrame) -> None:
    missing = REQUIRED_QUEUE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"review queue missing columns: {', '.join(sorted(missing))}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("review queue sample_id values must be non-empty and unique")


def load_queue(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    validate_queue(frame)
    return frame


def load_adjudications(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("decision") not in DECISIONS:
                raise ValueError(f"invalid decision at {path}:{line_number}")
            records[str(record["sample_id"])] = record
    return records


def save_adjudication(path: Path, record: dict[str, Any]) -> None:
    decision = record.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of: {', '.join(DECISIONS)}")
    if not record.get("sample_id"):
        raise ValueError("sample_id is required")
    records = load_adjudications(path)
    item = dict(record)
    item["sample_id"] = str(item["sample_id"])
    item["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    records[item["sample_id"]] = item
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(records):
            handle.write(json.dumps(records[sample_id], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def reviewed_count(queue: pd.DataFrame, records: dict[str, dict[str, Any]]) -> int:
    return int(queue["sample_id"].astype(str).isin(records).sum())
