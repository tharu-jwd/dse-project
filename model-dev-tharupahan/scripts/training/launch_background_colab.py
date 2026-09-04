#!/usr/bin/env python3
"""Launch one remote job detached from the Colab CLI execution request."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-script",
        type=Path,
        default=Path("/content/run_english_baseline_job_colab.py"),
    )
    parser.add_argument("--root", type=Path, default=Path("/content/sinhala-asr-job"))
    # IPython kernels add their own arguments when a file is injected by the
    # Colab CLI. They are unrelated to the job and are intentionally ignored.
    args, _ = parser.parse_known_args()
    output = args.root / "output"
    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / "remote-stdout.log"
    with stdout_path.open("ab", buffering=0) as stdout:
        process = subprocess.Popen(
            [sys.executable, str(args.job_script)],
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (output / "process.json").write_text(
        json.dumps({"pid": process.pid, "script": str(args.job_script)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pid": process.pid, "stdout": str(stdout_path)}))


if __name__ == "__main__":
    main()
