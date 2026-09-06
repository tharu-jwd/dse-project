#!/usr/bin/env python3
"""Stage the small, private E007 orchestration runtime for Kaggle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/kaggle/e007-orchestration-runtime"
SOURCES = (
    ROOT / "scripts/training/run_e002_colab.py",
    ROOT / "scripts/training/run_e007_kaggle.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    files = {}
    for source in SOURCES:
        destination = OUTPUT / source.name
        shutil.copy2(source, destination)
        files[destination.name] = {
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    metadata = {
        "title": "Sinhala ASR E007 Orchestration Runtime",
        "id": "tharupahan/sinhala-asr-e007-orchestration-runtime",
        "licenses": [{"name": "other"}],
        "description": "Private frozen E007 training orchestration scripts.",
    }
    (OUTPUT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "asset-index.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"directory": str(OUTPUT), "files": files}, indent=2))


if __name__ == "__main__":
    main()
