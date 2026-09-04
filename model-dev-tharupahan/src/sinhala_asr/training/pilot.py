"""Deterministic sampling for bounded adapter-training pilots."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Any


def _key(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _speaker_balanced(
    rows: list[dict[str, Any]], quota: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["speaker_id"])].append(row)
    queues = {
        speaker: deque(sorted(items, key=lambda row: _key(row["sample_id"], seed)))
        for speaker, items in groups.items()
    }
    speakers = sorted(queues, key=lambda speaker: _key(speaker, seed))
    selected: list[dict[str, Any]] = []
    while speakers and len(selected) < quota:
        remaining = []
        for speaker in speakers:
            selected.append(queues[speaker].popleft())
            if queues[speaker]:
                remaining.append(speaker)
            if len(selected) == quota:
                break
        speakers = remaining
    if len(selected) != quota:
        raise ValueError(f"requested {quota} rows but only selected {len(selected)}")
    return selected


def select_pilot_rows(
    rows: list[dict[str, Any]], *, total: int, latin_rows: int, seed: int
) -> list[dict[str, Any]]:
    if total <= 0 or latin_rows < 0 or latin_rows > total:
        raise ValueError("invalid pilot sample counts")
    train = [row for row in rows if row.get("dataset_split") == "train"]
    latin = [row for row in train if row.get("language_class") == "latin_only"]
    sinhala = [row for row in train if row.get("language_class") != "latin_only"]
    selected = _speaker_balanced(latin, latin_rows, seed)
    selected += _speaker_balanced(sinhala, total - latin_rows, seed)
    return sorted(selected, key=lambda row: _key(row["sample_id"], seed))
