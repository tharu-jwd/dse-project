#!/usr/bin/env python3
"""Freeze every row in the v4 training split as an ordered E007 selection."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/versions/v4/manifest.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pq.read_table(MANIFEST)
    train = manifest.filter(pc.equal(manifest.column("dataset_split"), "train"))
    sample_ids = sorted(train.column("sample_id").to_pylist())
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit("v4 training sample IDs are not unique")
    manifest_sha256 = sha256(MANIFEST)
    selection_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "dataset_split": "train",
                "sample_ids": sample_ids,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    result = {
        "experiment": "e007",
        "selection": "complete v4 training split ordered by sample_id",
        "manifest_sha256": manifest_sha256,
        "selection_fingerprint": selection_fingerprint,
        "rows": len(sample_ids),
        "hours": pc.sum(train.column("duration_seconds")).as_py() / 3600.0,
        "speakers": len(set(train.column("speaker_id").to_pylist())),
        "language_class_counts": dict(
            collections.Counter(train.column("language_class").to_pylist())
        ),
        "sample_ids": sample_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sample_ids"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
