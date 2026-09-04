#!/usr/bin/env python3
"""Run English retention evaluation for the original E001 LoRA adapter."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tarfile
import traceback
import urllib.request
from pathlib import Path

ROOT = Path("/content/sinhala-asr-job")
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
SOURCE = INPUT / "librispeech-test-clean-source.parquet"
BENCHMARK = INPUT / "e002-english-retention-librispeech-test-clean.parquet"
ADAPTER_ARCHIVE = Path("/content/e001-final-adapter.tar.gz")
ADAPTER = INPUT / "e001-final-adapter"
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
        event("job_started", source_url=SOURCE_URL, variant="e001")
        if not ADAPTER_ARCHIVE.exists():
            raise FileNotFoundError(ADAPTER_ARCHIVE)
        if not ADAPTER.exists():
            ADAPTER.mkdir(parents=True)
            with tarfile.open(ADAPTER_ARCHIVE, "r:gz") as archive:
                archive.extractall(ADAPTER, filter="data")
        for required in ("adapter_config.json", "adapter_model.safetensors"):
            if not (ADAPTER / required).is_file():
                raise FileNotFoundError(ADAPTER / required)
        event("adapter_extracted")
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
                "--adapter",
                str(ADAPTER),
                "--variant",
                "e001",
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
