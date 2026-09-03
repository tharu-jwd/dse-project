"""Manifest selection guards for Whisper prediction runs."""

from __future__ import annotations

from typing import Any


def select_prediction_rows(
    rows: list[dict[str, Any]],
    split: str,
    *,
    allow_unreviewed: bool = False,
    unlock_test: bool = False,
) -> list[dict[str, Any]]:
    if split.endswith("_candidate") and not allow_unreviewed:
        raise ValueError(
            "unreviewed candidate splits are allowed only for bounded smoke tests"
        )
    if split == "test" and not unlock_test:
        raise ValueError(
            "test prediction requires --unlock-test after the candidate is frozen"
        )
    selected = [row for row in rows if row.get("dataset_split") == split]
    if not selected:
        raise ValueError(f"manifest contains no rows for split: {split}")
    return selected
