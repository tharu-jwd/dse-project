#!/usr/bin/env python3
"""Monitor E002, sync complete checkpoints, and verify them locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def run(command: list[str], *, required: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if required and result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(session: str, remote: str, local: Path, *, required: bool = True) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        ["colab", "download", "-s", session, remote, str(local)], required=False
    )
    if result.returncode and required:
        raise RuntimeError(result.stderr or result.stdout)
    return result.returncode == 0


def record(path: Path, event: str, **fields: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    experiment = args.experiment_dir.expanduser().resolve()
    attempt = experiment / "attempts" / "training-001"
    archives = attempt / "checkpoint-downloads"
    checkpoints = experiment / "checkpoints"
    attempt.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    events = attempt / "operator-events.jsonl"
    index_path = attempt / "checkpoint-index-latest.json"

    while True:
        session_state = run(["colab", "status", "-s", args.session], required=False)
        if session_state.returncode:
            record(events, "session_unavailable", output=session_state.stderr)
            raise SystemExit("Colab session became unavailable")
        temporary_index = index_path.with_suffix(".json.partial")
        if download(
            args.session,
            "/content/sinhala-asr-job/output/checkpoint-index.json",
            temporary_index,
            required=False,
        ):
            temporary_index.replace(index_path)
            index = json.loads(index_path.read_text())
            for checkpoint in index["checkpoints"]:
                step = int(checkpoint["step"])
                destination = checkpoints / f"checkpoint-{step}"
                if destination.exists():
                    continue
                archive = archives / checkpoint["name"]
                partial = archive.with_suffix(archive.suffix + ".download.partial")
                download(
                    args.session,
                    f"/content/sinhala-asr-job/output/{checkpoint['name']}",
                    partial,
                )
                if sha256(partial) != checkpoint["sha256"]:
                    raise RuntimeError(f"checkpoint {step} download hash mismatch")
                partial.replace(archive)
                run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("verify_colab_checkpoint.py")),
                        "--archive",
                        str(archive),
                        "--sha256",
                        checkpoint["sha256"],
                        "--step",
                        str(step),
                        "--output-dir",
                        str(checkpoints),
                    ]
                )
                record(events, "checkpoint_verified", step=step, sha256=checkpoint["sha256"])
                print(f"verified checkpoint {step}", flush=True)

        status_path = attempt / "status.json"
        download(
            args.session,
            "/content/sinhala-asr-job/output/status.json",
            status_path,
            required=False,
        )
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        state = status.get("state")
        if state in {"complete", "failed"}:
            for name in (
                "events.jsonl",
                "remote-stdout.log",
                "resolved-config.json",
                "run-metadata.json",
                "sinhala-validation-predictions.parquet",
            ):
                download(
                    args.session,
                    f"/content/sinhala-asr-job/output/{name}",
                    attempt / name,
                    required=state == "complete",
                )
            record(events, "job_terminal", state=state)
            print(json.dumps(status, indent=2), flush=True)
            raise SystemExit(0 if state == "complete" else 1)
        print(f"job state: {state or 'unknown'}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
