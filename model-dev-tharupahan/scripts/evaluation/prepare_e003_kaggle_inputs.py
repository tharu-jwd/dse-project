#!/usr/bin/env python3
"""Build the private Kaggle input dataset for E003 English evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/kaggle/e003-english-evaluation-inputs"
BENCHMARK = ROOT / "reports/colab/e002-english-retention-librispeech-test-clean.parquet"
BENCHMARK_METADATA = ROOT / "reports/colab/e002-english-retention-librispeech-test-clean.json"
EXPERIMENT = ROOT / "reports/experiments/e003-english-replay-lora"
TRAINING = EXPERIMENT / "attempts/kaggle-training-002/output/e003-training"
ADAPTER = TRAINING / "final-adapter"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    benchmark = json.loads(BENCHMARK_METADATA.read_text())
    if sha256(BENCHMARK) != benchmark["bundle_sha256"]:
        raise SystemExit("English benchmark transport hash mismatch")
    if benchmark["rows"] != 2620:
        raise SystemExit("English benchmark row count mismatch")
    result = json.loads(
        (TRAINING.parent / "e003-kaggle-training-result.json").read_text()
    )
    if sha256(ADAPTER / "adapter_model.safetensors") != result["adapter_sha256"]:
        raise SystemExit("E003 adapter hash mismatch")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    shutil.copy2(BENCHMARK, OUTPUT / "english-retention-test-clean.parquet")
    shutil.copy2(BENCHMARK_METADATA, OUTPUT / "english-retention-metadata.json")
    shutil.copy2(
        ROOT / "scripts/evaluation/run_english_retention_colab.py",
        OUTPUT / "run_english_retention.py",
    )
    for source in ADAPTER.iterdir():
        if source.is_file():
            shutil.copy2(source, OUTPUT / f"adapter--{source.name}")
    metadata = {
        "title": "Sinhala ASR E003 Private English Evaluation Inputs",
        "id": "tharupahan/sinhala-asr-e003-english-evaluation-inputs",
        "licenses": [{"name": "other"}],
        "description": (
            "Private E003 adapter and frozen LibriSpeech test-clean benchmark. "
            "Benchmark audio is CC BY 4.0; adapter and code retain upstream terms."
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
