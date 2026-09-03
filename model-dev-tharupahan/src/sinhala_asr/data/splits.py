"""Deterministic speaker-disjoint candidate splits for native review."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any


def _choose_speakers(
    speakers: list[str], counts: Counter[str], target_rows: int, rng: random.Random
) -> set[str]:
    shuffled = list(speakers)
    rng.shuffle(shuffled)
    chosen: set[str] = set()
    rows = 0
    for speaker in shuffled:
        chosen.add(speaker)
        rows += counts[speaker]
        if rows >= target_rows:
            break
    return chosen


def make_speaker_disjoint_candidates(
    rows: list[dict[str, Any]], *, candidate_rows: int = 1000, seed: int = 20260903
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign train and held-out speaker pools; select review candidates within each pool."""
    valid = [row for row in rows if row.get("is_valid") and row.get("speaker_key")]
    counts = Counter(str(row["speaker_key"]) for row in valid)
    speakers = sorted(counts)
    rng = random.Random(seed)
    validation_speakers = _choose_speakers(speakers, counts, candidate_rows, rng)
    remaining = [speaker for speaker in speakers if speaker not in validation_speakers]
    test_speakers = _choose_speakers(remaining, counts, candidate_rows, rng)

    candidates: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    assigned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        speaker = item.get("speaker_key")
        if not item.get("is_valid"):
            item["dataset_split"] = "excluded"
            item["exclusion_reason"] = "automatic_validation_failure"
        elif not speaker:
            item["dataset_split"] = "excluded"
            item["exclusion_reason"] = "missing_speaker_identity"
        elif speaker in validation_speakers:
            candidates["validation"].append(item)
            item["dataset_split"] = "heldout_unused"
            item["exclusion_reason"] = "validation_speaker_pool"
        elif speaker in test_speakers:
            candidates["test"].append(item)
            item["dataset_split"] = "heldout_unused"
            item["exclusion_reason"] = "test_speaker_pool"
        else:
            item["dataset_split"] = "train"
            item["exclusion_reason"] = None
        assigned.append(item)

    for split, pool in candidates.items():
        pool.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['sample_id']}".encode()
            ).hexdigest()
        )
        selected = {row["sample_id"] for row in pool[:candidate_rows]}
        for item in assigned:
            if item["sample_id"] in selected:
                item["dataset_split"] = f"{split}_candidate"
                item["exclusion_reason"] = "requires_native_review"

    split_counts = Counter(item["dataset_split"] for item in assigned)
    fingerprint_input = "\n".join(
        f"{item['sample_id']}\t{item['dataset_split']}"
        for item in sorted(assigned, key=lambda row: row["sample_id"])
    )
    summary = {
        "seed": seed,
        "candidate_rows_requested_per_split": candidate_rows,
        "split_counts": dict(sorted(split_counts.items())),
        "validation_speakers": sorted(validation_speakers),
        "test_speakers": sorted(test_speakers),
        "speaker_disjoint": not bool(validation_speakers & test_speakers),
        "split_fingerprint": hashlib.sha256(fingerprint_input.encode()).hexdigest(),
    }
    return assigned, summary
