#!/usr/bin/env python3
"""Stage the pinned E016 evaluator as a small private Kaggle dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/evaluation/run_yohan_run7_kaggle.py"
OUTPUT = ROOT / "reports/kaggle/e016-runtime"


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
    destination = OUTPUT / SOURCE.name
    shutil.copy2(SOURCE, destination)
    metadata = {
        "title": "Sinhala ASR E016 Evaluation Runtime",
        "id": "tharupahan/sinhala-asr-e016-runtime",
        "licenses": [{"name": "other"}],
        "description": "Private pinned evaluator for Yohan run7 comparison.",
    }
    (OUTPUT / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "asset-index.json").write_text(
        json.dumps(
            {
                "files": {
                    destination.name: {
                        "bytes": destination.stat().st_size,
                        "sha256": sha256(destination),
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
