"""Deterministic stratified review-queue construction."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

from sinhala_asr.data.manifest import _audio_bytes


def _read_parquet_row(path: Path, row_index: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    offset = 0
    for group_index in range(parquet.num_row_groups):
        group_rows = parquet.metadata.row_group(group_index).num_rows
        if row_index < offset + group_rows:
            return parquet.read_row_group(group_index).slice(row_index - offset, 1).to_pylist()[0]
        offset += group_rows
    raise IndexError(f"row {row_index} is outside {path}")


def _first_present(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return None


def _pick(
    candidates: list[dict[str, Any]],
    quota: int,
    rng: random.Random,
    used: set[str],
) -> list[dict[str, Any]]:
    available = [row for row in candidates if row["sample_id"] not in used]
    rng.shuffle(available)
    chosen = available[:quota]
    used.update(row["sample_id"] for row in chosen)
    return chosen


def select_review_rows(rows: list[dict[str, Any]], quota: int = 100, seed: int = 20260903) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    used: set[str] = set()
    audio_counts = Counter(row.get("audio_pcm_sha256") or row.get("audio_sha256") for row in rows)
    categories: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("code_switched", lambda row: bool(row.get("is_code_switched"))),
        (
            "flagged_anomaly",
            lambda row: bool(json.loads(row.get("validation_flags") or "[]")),
        ),
        (
            "duplicate_candidate",
            lambda row: audio_counts[row.get("audio_pcm_sha256") or row.get("audio_sha256")] > 1,
        ),
        ("random_openslr", lambda row: row.get("source_dataset") == "openslr52"),
        (
            "random_collection",
            lambda row: row.get("source_dataset") in {"youtube", "bizbrains"},
        ),
    ]
    selected: list[dict[str, Any]] = []
    for category, predicate in categories:
        for row in _pick([row for row in rows if predicate(row)], quota, rng, used):
            item = dict(row)
            item["review_category"] = category
            selected.append(item)
    return selected


def attach_audio(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attached = []
    for row in rows:
        source_path = Path(row["source_path"])
        source_row = _read_parquet_row(source_path, int(row["source_row_index"]))
        value = _first_present(source_row, "audio", "audio_path", "file")
        item = dict(row)
        item["audio"] = _audio_bytes(value, source_path.parent)
        attached.append(item)
    return attached


def build_review_queue(manifest_path: Path, output_path: Path, quota: int = 100, seed: int = 20260903) -> pd.DataFrame:
    rows = pq.read_table(manifest_path).to_pylist()
    selected = attach_audio(select_review_rows(rows, quota=quota, seed=seed))
    frame = pd.DataFrame(selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, compression="zstd", index=False)
    return frame
