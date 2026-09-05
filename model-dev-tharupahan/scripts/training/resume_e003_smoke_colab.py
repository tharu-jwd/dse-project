#!/usr/bin/env python3
"""Install E003 smoke phase B config after phase A reaches checkpoint 1."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    poll = subprocess.run(
        [
            "colab",
            "exec",
            "-s",
            args.session,
            "-f",
            str(ROOT / "scripts/training/poll_background_colab.py"),
            "--timeout",
            "30",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = poll.stdout
    start = output.find("{")
    end = output.find("\nRECENT LOG OUTPUT")
    state = json.loads(output[start:end])
    status = state.get("status") or {}
    if state.get("alive") or status.get("state") != "complete":
        raise SystemExit("phase A is not complete")
    config = ROOT / "reports/colab/e003-smoke-phase-b-job-config.json"
    subprocess.run(
        [
            "colab",
            "upload",
            "-s",
            args.session,
            str(config),
            "/content/sinhala-asr-job/input/job-config.json",
        ],
        check=True,
    )
    subprocess.run(
        [
            "colab",
            "exec",
            "-s",
            args.session,
            "-f",
            str(ROOT / "scripts/training/launch_background_colab.py"),
            "--timeout",
            "30",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
