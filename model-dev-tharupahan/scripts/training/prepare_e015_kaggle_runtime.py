#!/usr/bin/env python3
"""Stage the exact E015 orchestration and training sources for Kaggle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/kaggle/e015-orchestration-runtime"
FILES = (
    ROOT / "scripts/training/run_e015_kaggle.py",
    ROOT / "scripts/training/run_e015_kaggle_smoke.py",
    ROOT / "scripts/training/train.py",
    ROOT / "scripts/evaluation/predict.py",
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
    for source in FILES:
        shutil.copy2(source, OUTPUT / source.name)
    shutil.copytree(
        ROOT / "src/sinhala_asr",
        OUTPUT / "sinhala_asr",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    metadata = {
        "title": "Sinhala ASR E015 Orchestration Runtime",
        "id": "tharupahan/sinhala-asr-e015-orchestration-runtime",
        "licenses": [{"name": "other"}],
        "description": (
            "Private frozen E015 full-parameter training, prediction, and "
            "resumable orchestration sources."
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
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "asset-index.json"
    }
    (OUTPUT / "asset-index.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"directory": str(OUTPUT), "files": files}, indent=2))


if __name__ == "__main__":
    main()
