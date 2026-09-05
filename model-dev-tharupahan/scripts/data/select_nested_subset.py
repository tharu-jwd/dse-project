#!/usr/bin/env python3
"""Select a deterministic, speaker-balanced nested v4 training subset.

Builds a superset of a smaller frozen subset (identified by its exact
sample IDs) by adding rows from the remaining v4 training pool until a
target total duration is reached. Selection order within each language
class is a deterministic per-speaker round robin: speakers are visited in
sorted order, each speaker's own remaining rows are shuffled with a fixed
seed, and one row per speaker is taken per round. This spreads the added
rows across as many speakers as possible before any single speaker
contributes a second row, matching the "speaker-balanced" selection
already used for the frozen smaller subsets.

This script only selects sample IDs and reports statistics; it does not
write shard Parquet files. Use it to freeze a manifest of IDs, then stage
shards from that manifest the same way prior experiments did.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/versions/v4/manifest.parquet"


def load_frozen_ids(shard_glob: str) -> set[str]:
    ids: set[str] = set()
    for path in sorted(Path().glob(shard_glob)):
        table = pq.read_table(path, columns=["sample_id"])
        ids.update(table.column("sample_id").to_pylist())
    if not ids:
        raise SystemExit(f"no rows found for shard glob: {shard_glob}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-shard-glob", required=True)
    parser.add_argument("--target-hours", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen_ids = load_frozen_ids(args.frozen_shard_glob)

    manifest = pq.read_table(MANIFEST)
    train = manifest.filter(pc.equal(manifest.column("dataset_split"), "train"))
    train_ids = set(train.column("sample_id").to_pylist())
    if not frozen_ids.issubset(train_ids):
        raise SystemExit("frozen shard IDs are not a subset of the v4 train split")

    frozen_mask = pc.is_in(train.column("sample_id"), value_set=pa.array(frozen_ids))
    frozen_rows = train.filter(frozen_mask)
    frozen_hours = pc.sum(frozen_rows.column("duration_seconds")).as_py() / 3600.0

    remaining_mask = pc.invert(frozen_mask)
    remaining = train.filter(remaining_mask)

    target_additional_seconds = max(0.0, args.target_hours - frozen_hours) * 3600.0

    columns = remaining.select(["sample_id", "speaker_id", "duration_seconds"]).to_pylist()
    by_speaker: dict[str, list[dict]] = collections.defaultdict(list)
    for row in columns:
        by_speaker[row["speaker_id"]].append(row)

    rng = random.Random(args.seed)
    for speaker_id in by_speaker:
        rng.shuffle(by_speaker[speaker_id])

    queues = {speaker_id: iter(rows) for speaker_id, rows in by_speaker.items()}
    active = sorted(by_speaker)

    selected: list[dict] = []
    selected_seconds = 0.0
    while active and selected_seconds < target_additional_seconds:
        next_active = []
        for speaker_id in active:
            if selected_seconds >= target_additional_seconds:
                next_active.append(speaker_id)
                continue
            try:
                row = next(queues[speaker_id])
            except StopIteration:
                continue
            selected.append(row)
            selected_seconds += row["duration_seconds"]
            next_active.append(speaker_id)
        active = next_active

    added_ids = {row["sample_id"] for row in selected}
    final_ids = frozen_ids | added_ids

    final_mask = pc.is_in(train.column("sample_id"), value_set=pa.array(final_ids))
    final_rows = train.filter(final_mask)
    final_hours = pc.sum(final_rows.column("duration_seconds")).as_py() / 3600.0
    final_speakers = len(set(final_rows.column("speaker_id").to_pylist()))
    language_counts = collections.Counter(final_rows.column("language_class").to_pylist())

    fingerprint_input = json.dumps(
        {
            "manifest_fingerprint": hashlib.sha256(
                MANIFEST.read_bytes()
            ).hexdigest(),
            "frozen_ids": sorted(frozen_ids),
            "added_ids": sorted(added_ids),
            "seed": args.seed,
        },
        sort_keys=True,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_input).hexdigest()

    result = {
        "frozen_rows": len(frozen_ids),
        "frozen_hours": frozen_hours,
        "added_rows": len(added_ids),
        "final_rows": len(final_ids),
        "final_hours": final_hours,
        "final_speakers": final_speakers,
        "language_class_counts": dict(language_counts),
        "seed": args.seed,
        "selection_fingerprint": fingerprint,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**result, "sample_ids": sorted(final_ids)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
