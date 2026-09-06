#!/usr/bin/env python3
"""Stage the complete 182,665-row v4 training split for private Kaggle use."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARDS = ROOT / "reports/e007-preparation/sinhala-shards"
OUTPUT = ROOT / "reports/kaggle/e007-input-dataset"
VALIDATION = ROOT / "reports/kaggle/e003-input-dataset/v4-validation-206-audio.parquet"
EXPECTED_ROWS = 182_665
EXPECTED_VALIDATION_SHA256 = "c7a378a115dd953c50bfe6a2c550e28b5a5cc829dee79bdb29ccde1151e46ec2"
EXPECTED_SHARDS_COMBINED_SHA256 = (
    "ecd5bbffee8a644fd2db83385799838ca055cc832335aaa90de00ca7db35e6f9"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    shard_manifest = json.loads((SHARDS / "manifest.json").read_text())
    if shard_manifest["rows"] != EXPECTED_ROWS:
        raise SystemExit("E007 shard row count mismatch")
    if shard_manifest["shards_combined_sha256"] != EXPECTED_SHARDS_COMBINED_SHA256:
        raise SystemExit("E007 shard combined-hash mismatch")
    if sha256(VALIDATION) != EXPECTED_VALIDATION_SHA256:
        raise SystemExit("validation bundle hash mismatch")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    staged_shards = []
    total_rows = 0
    for shard in shard_manifest["shards"]:
        source = SHARDS / shard["name"]
        if sha256(source) != shard["sha256"]:
            raise SystemExit(f"source shard hash mismatch: {source.name}")
        name = f"sinhala-{shard['name']}"
        destination = OUTPUT / name
        link(source, destination)
        staged_shards.append({**shard, "name": name})
        total_rows += shard["rows"]
    if total_rows != EXPECTED_ROWS:
        raise SystemExit("staged row count mismatch")
    link(VALIDATION, OUTPUT / VALIDATION.name)
    manifest = {
        "experiment": "e007",
        "dataset_fingerprint": shard_manifest["dataset_fingerprint"],
        "selection_fingerprint": shard_manifest["selection_fingerprint"],
        "selection": "complete v4 training split ordered by sample_id",
        "seed": None,
        "rows": EXPECTED_ROWS,
        "hours": 220.87716666666668,
        "speakers": 471,
        "language_class_counts": {"sinhala_only": 175886, "latin_only": 6779},
        "shards_combined_sha256": EXPECTED_SHARDS_COMBINED_SHA256,
        "shards": staged_shards,
    }
    (OUTPUT / "sinhala-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "title": "Sinhala ASR E007 Private Full v4 Training Inputs",
        "id": "tharupahan/sinhala-asr-e007-inputs",
        "licenses": [{"name": "other"}],
        "description": (
            "Private complete 182,665-row, 220.877-hour v4 training split and "
            "frozen validation bundle; source data is OpenSLR 52."
        ),
    }
    (OUTPUT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    files = {
        str(path.relative_to(OUTPUT)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "asset-index.json"
    }
    (OUTPUT / "asset-index.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "directory": str(OUTPUT),
                "files": len(files) + 1,
                "rows": total_rows,
                "bytes": sum(value["bytes"] for value in files.values()),
                "shards_combined_sha256": EXPECTED_SHARDS_COMBINED_SHA256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
