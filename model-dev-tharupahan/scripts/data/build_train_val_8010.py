#!/usr/bin/env python3
"""Move whole speakers from v4 train into validation to hit ~80/10/10.

Fixed points: total OpenSLR (train+validation) = 224.475h; test is fixed
externally at 22.72h (TTS + YouTube, see build_external_eval_manifest.py).
Target: validation ~= 10% of the grand total (train+validation+test), which
means pulling ~21.1 additional hours of train speakers into validation on
top of the existing 3.598h (392 reviewed + 2,230 unreviewed held-out rows,
see build_openslr_validation_v2.py).

Selection is deterministic: train speakers sorted by speaker_id, whole
speakers moved in order until the target additional hours is reached (never
split a speaker across train and validation). v4's own manifest.parquet is
NOT modified -- this writes two new additive manifests.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

V4_MANIFEST = "data/versions/v4/manifest.parquet"
EXISTING_VALIDATION = "reports/dataset-audit/openslr-validation-v2/manifest.parquet"
TRAIN_OUTPUT = "reports/dataset-audit/openslr-train-v2/manifest.parquet"
VALIDATION_OUTPUT = "reports/dataset-audit/openslr-validation-v2/manifest.parquet"

TEST_HOURS = 22.72  # TTS + YouTube, fixed
TARGET_VALIDATION_SHARE = 0.10


def main() -> None:
    df = pd.read_parquet(V4_MANIFEST)
    train = df[df["dataset_split"] == "train"].copy()
    existing_validation = pd.read_parquet(EXISTING_VALIDATION)

    openslr_total_hours = train["duration_seconds"].sum() / 3600 + existing_validation[
        "duration_seconds"
    ].sum() / 3600
    grand_total_hours = openslr_total_hours + TEST_HOURS
    target_validation_hours = grand_total_hours * TARGET_VALIDATION_SHARE
    additional_needed_hours = target_validation_hours - existing_validation[
        "duration_seconds"
    ].sum() / 3600
    print(f"OpenSLR total: {openslr_total_hours:.2f}h; grand total: {grand_total_hours:.2f}h")
    print(f"target validation: {target_validation_hours:.2f}h; need +{additional_needed_hours:.2f}h from train")

    per_speaker = (
        train.groupby("speaker_id")["duration_seconds"].sum().sort_index() / 3600
    )
    moved_speakers: list[str] = []
    accumulated = 0.0
    for speaker_id, hours in per_speaker.items():
        if accumulated >= additional_needed_hours:
            break
        moved_speakers.append(speaker_id)
        accumulated += hours

    print(f"moving {len(moved_speakers)} speakers, {accumulated:.3f}h")

    moved_rows = train[train["speaker_id"].isin(moved_speakers)].copy()
    moved_rows["dataset_split"] = "validation"
    moved_rows["is_audio_reviewed"] = False  # never audio-reviewed as held-out material

    remaining_train = train[~train["speaker_id"].isin(moved_speakers)].copy()
    new_validation = pd.concat([existing_validation, moved_rows], ignore_index=True)
    new_validation = new_validation.sort_values("sample_id").reset_index(drop=True)

    train_fp = hashlib.sha256(
        "\n".join(sorted(remaining_train["sample_id"].astype(str))).encode("utf-8")
    ).hexdigest()
    val_fp = hashlib.sha256(
        "\n".join(sorted(new_validation["sample_id"].astype(str))).encode("utf-8")
    ).hexdigest()

    pq.write_table(pa.Table.from_pandas(remaining_train, preserve_index=False), TRAIN_OUTPUT, compression="zstd")
    pq.write_table(pa.Table.from_pandas(new_validation, preserve_index=False), VALIDATION_OUTPUT, compression="zstd")

    final_train_hours = remaining_train["duration_seconds"].sum() / 3600
    final_val_hours = new_validation["duration_seconds"].sum() / 3600
    final_total = final_train_hours + final_val_hours + TEST_HOURS
    print()
    print(f"train-v2: {len(remaining_train)} rows, {final_train_hours:.2f}h, "
          f"{remaining_train['speaker_id'].nunique()} speakers  ({final_train_hours/final_total:.1%})  fp={train_fp}")
    print(f"validation-v2: {len(new_validation)} rows, {final_val_hours:.2f}h, "
          f"{new_validation['speaker_id'].nunique()} speakers  ({final_val_hours/final_total:.1%})  fp={val_fp}")
    print(f"test (unchanged): {TEST_HOURS:.2f}h  ({TEST_HOURS/final_total:.1%})")
    print(f"grand total: {final_total:.2f}h")


if __name__ == "__main__":
    main()
