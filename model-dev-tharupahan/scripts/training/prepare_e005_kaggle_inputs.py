#!/usr/bin/env python3
"""Stage the private 50k-row E005 Sinhala scale dataset for Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/e005-preparation"
SHARDS = SOURCE / "sinhala-shards"
OUTPUT = ROOT / "reports/kaggle/e005-input-dataset"
VALIDATION = ROOT / "reports/kaggle/e003-input-dataset/v4-validation-206-audio.parquet"
EXPECTED_SOURCE_SHA256 = "00fae26f174f6abb9f360ea9b91d064f17cf8ec609fe44d9375a2d94d8df104a"
EXPECTED_VALIDATION_SHA256 = "c7a378a115dd953c50bfe6a2c550e28b5a5cc829dee79bdb29ccde1151e46ec2"


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
    source_metadata = json.loads((SOURCE / "e005-v4-50000.json").read_text())
    shard_manifest = json.loads((SHARDS / "manifest.json").read_text())
    if source_metadata["bundle_sha256"] != EXPECTED_SOURCE_SHA256:
        raise SystemExit("E005 source bundle hash mismatch")
    if shard_manifest["source_sha256"] != EXPECTED_SOURCE_SHA256:
        raise SystemExit("E005 shard source hash mismatch")
    if source_metadata["rows"] != 50000 or shard_manifest["rows"] != 50000:
        raise SystemExit("E005 source row count mismatch")
    if sha256(VALIDATION) != EXPECTED_VALIDATION_SHA256:
        raise SystemExit("validation bundle hash mismatch")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    staged_shards = []
    for shard in shard_manifest["shards"]:
        source = SHARDS / shard["name"]
        if sha256(source) != shard["sha256"]:
            raise SystemExit(f"source shard hash mismatch: {source.name}")
        name = f"sinhala-{shard['name']}"
        destination = OUTPUT / name
        link(source, destination)
        staged_shards.append({**shard, "name": name})
    link(VALIDATION, OUTPUT / VALIDATION.name)
    manifest = {
        "experiment": "e005",
        "dataset_fingerprint": source_metadata["dataset_fingerprint"],
        "selection": source_metadata["selection"],
        "seed": source_metadata["seed"],
        "rows": 50000,
        "latin_rows": source_metadata["latin_rows"],
        "speakers": source_metadata["speakers"],
        "hours": source_metadata["hours"],
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "shards": staged_shards,
    }
    (OUTPUT / "sinhala-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "title": "Sinhala ASR E005 Private 50k Inputs",
        "id": "tharupahan/sinhala-asr-e005-inputs",
        "licenses": [{"name": "other"}],
        "description": (
            "Private 50,000-row deterministic v4 Sinhala-source training subset "
            "and frozen validation bundle; source data is OpenSLR 52."
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
    print(json.dumps({"directory": str(OUTPUT), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
