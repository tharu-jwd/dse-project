#!/usr/bin/env python3
"""Verify and stage all E003 inputs into an already allocated Colab session."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLAB = ROOT / "reports" / "colab"
MIXED = COLAB / "e003-train-mixed-shards" / "manifest.json"
E002 = COLAB / "e002-train-v4-10000-shards"
REPLAY = COLAB / "e003-english-replay-shards"


def run(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def upload(session: str, local: Path, remote: str) -> None:
    run(["colab", "upload", "-s", session, str(local), remote])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=COLAB / "e003-smoke-phase-a-job-config.json",
    )
    args = parser.parse_args()
    manifest = json.loads(MIXED.read_text())
    for shard in manifest["shards"]:
        prefix, name = shard["name"].split("/", 1)
        local = (E002 if prefix == "e002" else REPLAY) / name
        if local.stat().st_size != shard["bytes"] or digest(local) != shard["sha256"]:
            raise SystemExit(f"local shard verification failed: {shard['name']}")

    run(["colab", "status", "-s", args.session])
    run(
        [
            "colab",
            "exec",
            "-s",
            args.session,
            "-f",
            str(ROOT / "scripts/training/prepare_colab_workspace.py"),
            "--timeout",
            "60",
        ]
    )
    upload(args.session, MIXED, "/content/sinhala-asr-job/input/train/manifest.json")
    for shard in manifest["shards"]:
        prefix, name = shard["name"].split("/", 1)
        local = (E002 if prefix == "e002" else REPLAY) / name
        upload(
            args.session,
            local,
            f"/content/sinhala-asr-job/input/train/{prefix}/{name}",
        )
    upload(
        args.session,
        COLAB / "v4-validation-206-audio.parquet",
        "/content/sinhala-asr-job/input/v4-validation-206-audio.parquet",
    )
    upload(args.session, args.config, "/content/sinhala-asr-job/input/job-config.json")
    for name in (
        "run_e002_colab.py",
        "run_e003_colab.py",
        "launch_background_colab.py",
        "poll_background_colab.py",
    ):
        upload(
            args.session,
            ROOT / "scripts" / "training" / name,
            f"/content/{name}",
        )
    print("E003 inputs staged and hash-verified; job has not been launched.")


if __name__ == "__main__":
    main()
