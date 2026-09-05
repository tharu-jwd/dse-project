#!/usr/bin/env python3
"""Build a private offline Kaggle runtime with wheels and Whisper-small."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/kaggle/e003-runtime-dataset"
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--openai--whisper-small/snapshots"
    / "973afd24965f72e36ca33b3055d56a652f456b4d"
)
PACKAGES = (
    "transformers==5.16.1",
    "peft==0.20.0",
    "huggingface-hub==1.30.0",
    "tokenizers==0.23.2",
    "safetensors==0.8.0",
    "accelerate==1.14.0",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not (SNAPSHOT / "model.safetensors").is_file():
        raise SystemExit(f"complete Whisper-small snapshot missing: {SNAPSHOT}")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(OUTPUT),
            "--only-binary=:all:",
            "--no-deps",
            "--platform=manylinux2014_x86_64",
            "--python-version=312",
            *PACKAGES,
        ],
        check=True,
    )
    for source in SNAPSHOT.iterdir():
        if source.is_file():
            shutil.copy2(
                source,
                OUTPUT / f"whisper-small--{source.name}",
                follow_symlinks=True,
            )
    for name in ("run_e002_colab.py", "run_e003_kaggle_smoke.py"):
        shutil.copy2(ROOT / "scripts/training" / name, OUTPUT / name)
    metadata = {
        "title": "Sinhala ASR E003 Private Offline Runtime",
        "id": "tharupahan/sinhala-asr-e003-runtime",
        "licenses": [{"name": "other"}],
        "description": (
            "Private offline experiment runtime: OpenAI Whisper-small model files, "
            "Python wheels, and project runners. Subject to upstream licenses."
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
