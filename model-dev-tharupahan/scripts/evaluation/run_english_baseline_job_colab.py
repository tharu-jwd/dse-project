#!/usr/bin/env python3
"""Run the hash-pinned English baseline job and persist inspectable status."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import traceback
import urllib.request
from pathlib import Path

ROOT = Path("/content/sinhala-asr-job")
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SOURCE = INPUT / "librispeech-test-clean-source.parquet"
BENCHMARK = INPUT / "e002-english-retention-librispeech-test-clean.parquet"
STATUS = OUTPUT / "status.json"
EVENTS = OUTPUT / "events.jsonl"
SOURCE_URL = (
    "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/"
    "main/all/test.clean/0000.parquet"
)


def event(kind: str, **fields: object) -> None:
    record = {
        "time_utc": dt.datetime.now(dt.UTC).isoformat(),
        "event": kind,
        **fields,
    }
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def command(arguments: list[str]) -> None:
    event("command_started", arguments=arguments)
    subprocess.run(arguments, check=True)
    event("command_completed", arguments=arguments)


def main() -> None:
    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"state": "running"}) + "\n", encoding="utf-8")
    try:
        event("job_started", source_url=SOURCE_URL)
        if not SOURCE.exists():
            urllib.request.urlretrieve(SOURCE_URL, SOURCE)
            event("source_downloaded", bytes=SOURCE.stat().st_size)
        command(
            [
                sys.executable,
                "/content/prepare_english_retention_benchmark.py",
                "--source",
                str(SOURCE),
                "--output",
                str(BENCHMARK),
            ]
        )
        command(
            [
                sys.executable,
                "/content/run_english_retention_colab.py",
                "--benchmark",
                str(BENCHMARK),
                "--output-dir",
                str(OUTPUT),
                "--variant",
                "untouched",
            ]
        )
        status = {"state": "complete", "exit_code": 0}
        event("job_completed")
    except Exception as error:
        status = {
            "state": "failed",
            "exit_code": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        event("job_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
