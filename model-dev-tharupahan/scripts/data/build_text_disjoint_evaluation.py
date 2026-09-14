#!/usr/bin/env python3
"""Derive transcript-disjoint validation and test manifests without changing train.

Dataset v5 is audio- and speaker-disjoint, but many OpenSLR prompts occur in
both train and validation with different speakers. This script creates the v6
evaluation views used for model selection and final reporting. Source manifests
remain immutable, and excluded rows are preserved as audit artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_values(frame: pd.DataFrame) -> set[str]:
    if "text_metric" not in frame.columns:
        raise ValueError("manifest is missing text_metric")
    text = frame["text_metric"]
    if text.isna().any() or text.astype(str).str.strip().eq("").any():
        raise ValueError("text_metric contains null or empty values")
    return set(text.astype(str))


def assert_disjoint(left: pd.DataFrame, right: pd.DataFrame, names: tuple[str, str]) -> None:
    for column in ("sample_id", "audio_sha256", "audio_pcm_sha256", "text_metric"):
        if column not in left.columns or column not in right.columns:
            raise ValueError(f"required comparison column is missing: {column}")
        overlap = set(left[column].dropna().astype(str)) & set(
            right[column].dropna().astype(str)
        )
        if overlap:
            raise ValueError(
                f"{names[0]}/{names[1]} overlap in {column}: {len(overlap)} values"
            )


def stats(frame: pd.DataFrame) -> dict[str, object]:
    reviewed = (
        int(frame["is_audio_reviewed"].fillna(False).astype(bool).sum())
        if "is_audio_reviewed" in frame.columns
        else None
    )
    return {
        "rows": len(frame),
        "hours": float(frame["duration_seconds"].sum() / 3600),
        "speaker_ids": int(frame["speaker_id"].nunique()),
        "uploaders": int(frame["uploader"].nunique()),
        "audio_reviewed_rows": reviewed,
        "language_classes": {
            str(key): int(value)
            for key, value in frame["language_class"].value_counts().items()
        },
    }


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False),
        path,
        compression="zstd",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("reports/dataset-audit/openslr-train-v2/manifest.parquet"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("reports/dataset-audit/openslr-validation-v2/manifest.parquet"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("reports/dataset-audit/external-eval-v1/manifest.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dataset-audit/v6-text-disjoint"),
    )
    args = parser.parse_args()

    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    test = pd.read_parquet(args.test)

    train_text = text_values(train)
    validation_overlap = validation["text_metric"].astype(str).isin(train_text)
    clean_validation = validation.loc[~validation_overlap].copy()
    excluded_validation = validation.loc[validation_overlap].copy()

    forbidden_test_text = train_text | text_values(clean_validation)
    test_overlap = test["text_metric"].astype(str).isin(forbidden_test_text)
    clean_test = test.loc[~test_overlap].copy()
    excluded_test = test.loc[test_overlap].copy()

    assert_disjoint(train, clean_validation, ("train", "validation"))
    assert_disjoint(train, clean_test, ("train", "test"))
    assert_disjoint(clean_validation, clean_test, ("validation", "test"))

    outputs = {
        "validation": args.output_dir / "validation-manifest.parquet",
        "test": args.output_dir / "test-manifest.parquet",
        "excluded_validation": args.output_dir / "excluded-validation-text-overlap.parquet",
        "excluded_test": args.output_dir / "excluded-test-text-overlap.parquet",
    }
    write_parquet(clean_validation, outputs["validation"])
    write_parquet(clean_test, outputs["test"])
    write_parquet(excluded_validation, outputs["excluded_validation"])
    write_parquet(excluded_test, outputs["excluded_test"])

    summary = {
        "dataset_version": "v6-text-disjoint-evaluation",
        "comparison_field": "text_metric",
        "source_files": {
            "train": {"path": str(args.train), "sha256": file_sha256(args.train)},
            "validation": {
                "path": str(args.validation),
                "sha256": file_sha256(args.validation),
            },
            "test": {"path": str(args.test), "sha256": file_sha256(args.test)},
        },
        "train_unchanged": stats(train),
        "validation": stats(clean_validation),
        "test": stats(clean_test),
        "excluded_for_cross_split_text_overlap": {
            "validation_rows": len(excluded_validation),
            "test_rows": len(excluded_test),
        },
        "outputs": {},
        "verified_pairwise_overlap": {
            "sample_id": 0,
            "audio_sha256": 0,
            "audio_pcm_sha256": 0,
            "text_metric": 0,
        },
    }
    summary["outputs"] = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in outputs.items()
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
