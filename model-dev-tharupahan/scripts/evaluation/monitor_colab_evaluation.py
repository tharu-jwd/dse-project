#!/usr/bin/env python3
"""Monitor one background Colab evaluation and retrieve terminal artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def download(session: str, remote: str, local: Path, required: bool) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["colab", "download", "-s", session, remote, str(local)],
        capture_output=True,
        text=True,
    )
    if result.returncode and required:
        raise RuntimeError(result.stderr or result.stdout)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    while True:
        session = subprocess.run(
            ["colab", "status", "-s", args.session], capture_output=True, text=True
        )
        if session.returncode:
            raise SystemExit("Colab session became unavailable")
        status_path = output / "status.json"
        download(
            args.session,
            "/content/sinhala-asr-job/output/status.json",
            status_path,
            False,
        )
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        state = status.get("state")
        print(f"evaluation state: {state or 'unknown'}", flush=True)
        if state in {"complete", "failed"}:
            for name in ("events.jsonl", "remote-stdout.log", "process.json"):
                download(
                    args.session,
                    f"/content/sinhala-asr-job/output/{name}",
                    output / name,
                    True,
                )
            if state == "complete":
                for name in (
                    f"{args.variant}-predictions.parquet",
                    f"{args.variant}-runtime.json",
                ):
                    download(
                        args.session,
                        f"/content/sinhala-asr-job/output/{name}",
                        output / name,
                        True,
                    )
            raise SystemExit(0 if state == "complete" else 1)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
