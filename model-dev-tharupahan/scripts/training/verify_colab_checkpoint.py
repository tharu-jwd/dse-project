#!/usr/bin/env python3
"""Verify and atomically install a downloaded Colab checkpoint archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

REQUIRED = {
    "COMPLETE",
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    actual_hash = sha256(archive)
    if actual_hash != args.sha256:
        raise SystemExit(f"archive hash mismatch: {actual_hash}")
    destination = args.output_dir.expanduser().resolve() / f"checkpoint-{args.step}"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(staging, filter="data")
        roots = [path for path in staging.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise SystemExit("checkpoint archive must contain exactly one directory")
        checkpoint = roots[0]
        missing = sorted(REQUIRED - {path.name for path in checkpoint.iterdir()})
        if missing:
            raise SystemExit(f"checkpoint files missing: {missing}")
        marker_step = int((checkpoint / "COMPLETE").read_text().strip())
        trainer_step = int(json.loads((checkpoint / "trainer_state.json").read_text())["global_step"])
        if marker_step != args.step or trainer_step != args.step:
            raise SystemExit(
                f"checkpoint step mismatch: {marker_step}/{trainer_step}/{args.step}"
            )
        shutil.move(checkpoint, destination)
    latest = {
        "step": args.step,
        "checkpoint": str(destination),
        "archive_sha256": actual_hash,
    }
    latest_path = args.output_dir / "latest-verified.json"
    partial = latest_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
    partial.replace(latest_path)
    print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    main()
