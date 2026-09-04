#!/usr/bin/env python3
"""Build a self-contained, risk-stratified boundary-crop review queue."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf


def crop_wav(path: str, start: float, end: float) -> bytes:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    cropped = samples[round(start * sample_rate) : round(end * sample_rate)]
    output = io.BytesIO()
    sf.write(output, cropped, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def select_rows(frame: pd.DataFrame, quota: int, seed: int) -> pd.DataFrame:
    """Select across trim-severity quantiles, including the most aggressive crops."""
    if quota >= len(frame):
        return frame.copy()
    ranked = frame.sort_values("saved_seconds", ascending=False)
    extreme_count = min(max(10, quota // 5), quota)
    selected = ranked.head(extreme_count)
    remaining = frame.drop(selected.index).copy()
    remaining["severity_bin"] = pd.qcut(
        remaining["saved_seconds"],
        min(5, remaining.saved_seconds.nunique()),
        duplicates="drop",
    )
    needed = quota - len(selected)
    sampled = remaining.groupby("severity_bin", observed=True, group_keys=False).apply(
        lambda group: group.sample(
            n=min(len(group), max(1, needed // 5)), random_state=seed
        ),
        include_groups=False,
    )
    selected = pd.concat([selected, sampled]).drop_duplicates("sample_id")
    if len(selected) < quota:
        fill = remaining.drop(selected.index, errors="ignore").sample(
            n=quota - len(selected), random_state=seed
        )
        selected = pd.concat([selected, fill])
    return selected.head(quota).sort_values(
        ["saved_seconds", "sample_id"], ascending=[False, True]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    manifest = pd.read_parquet(args.manifest.expanduser().resolve())
    proposals = pd.read_parquet(args.proposals.expanduser().resolve())
    frame = manifest.merge(proposals, on="sample_id", validate="one_to_one")
    frame = frame[frame.dataset_split.isin(["train", "validation", "test"])].copy()
    chosen = select_rows(frame, args.quota, args.seed)
    source_tables: dict[str, list[dict]] = {}
    output_rows = []
    for row in chosen.to_dict("records"):
        source_path = str(row["source_path"])
        if source_path not in source_tables:
            source_tables[source_path] = pq.read_table(
                source_path, columns=["audio"]
            ).to_pylist()
        audio_path = source_tables[source_path][int(row["source_row_index"])]["audio"]
        if not isinstance(audio_path, str):
            raise ValueError("crop review currently requires path-backed audio")
        original = Path(audio_path).read_bytes()
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "text_original": row["text_canonical"],
                "dataset_split": row["dataset_split"],
                "original_duration_seconds": row["original_duration_seconds"],
                "crop_start_seconds": row["crop_start_seconds"],
                "crop_end_seconds": row["crop_end_seconds"],
                "saved_seconds": row["saved_seconds"],
                "original_audio": original,
                "cropped_audio": crop_wav(
                    audio_path, row["crop_start_seconds"], row["crop_end_seconds"]
                ),
            }
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_parquet(output, compression="zstd", index=False)
    print(f"Wrote {len(output_rows)} crop-review samples to {output}")


if __name__ == "__main__":
    main()
