#!/usr/bin/env python3
"""Assemble hash-verified private Kaggle dataset assets for E003."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLAB = ROOT / "reports" / "colab"
OUTPUT = ROOT / "reports" / "kaggle" / "e003-input-dataset"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(source: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mixed_path = COLAB / "e003-train-mixed-shards/manifest.json"
    mixed = json.loads(mixed_path.read_text())
    flat_shards = []
    for shard in mixed["shards"]:
        group, filename = shard["name"].split("/", 1)
        source_dir = (
            COLAB / "e002-train-v4-10000-shards"
            if group == "e002"
            else COLAB / "e003-english-replay-shards"
        )
        source = source_dir / filename
        target_name = f"{group}-{filename}"
        target = OUTPUT / target_name
        if source.stat().st_size != shard["bytes"] or sha256(source) != shard["sha256"]:
            raise SystemExit(f"source shard verification failed: {source}")
        link(source, target)
        flat_shards.append({**shard, "name": target_name})
    kaggle_manifest = {**mixed, "shards": flat_shards}
    (OUTPUT / "train-manifest.json").write_text(
        json.dumps(kaggle_manifest, indent=2) + "\n", encoding="utf-8"
    )
    assets = {
        "v4-validation-206-audio.parquet": COLAB
        / "v4-validation-206-audio.parquet",
        "e003-smoke-phase-a-job-config.json": COLAB
        / "e003-smoke-phase-a-job-config.json",
        "e003-smoke-phase-b-job-config.json": COLAB
        / "e003-smoke-phase-b-job-config.json",
        "e003-job-config.json": COLAB / "e003-job-config.json",
        "run_e002_colab.py": ROOT / "scripts/training/run_e002_colab.py",
        "run_e003_kaggle_smoke.py": ROOT
        / "scripts/training/run_e003_kaggle_smoke.py",
    }
    for name, source in assets.items():
        link(source, OUTPUT / name)
    metadata = {
        "title": "Sinhala ASR E003 Private Training Inputs",
        "id": "tharupahan/sinhala-asr-e003-inputs",
        "licenses": [{"name": "unknown"}],
        "description": (
            "Private, immutable E003 experiment transport. Contains project-owned "
            "Sinhala data and a CC BY 4.0 LibriSpeech replay subset. Not for publication."
        ),
    }
    (OUTPUT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    record = {
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(OUTPUT.iterdir())
            if path.name != "asset-index.json"
        }
    }
    (OUTPUT / "asset-index.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"directory": str(OUTPUT), **record}, indent=2))


if __name__ == "__main__":
    main()
