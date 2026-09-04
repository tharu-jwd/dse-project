#!/usr/bin/env python3
"""Validate and freeze the official LibriSpeech test-clean English benchmark."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

EXPECTED_SOURCE_SHA256 = (
    "7113aa4c3cf963fb54697145719a7725f984c8836d1c494a554cbb9f1a017df0"
)
EXPECTED_ROWS = 2620


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_fingerprint(rows: list[dict]) -> str:
    """Hash semantic row content independently of Parquet serialization."""
    digest = hashlib.sha256()
    for row in rows:
        identity = {
            "sample_id": row["sample_id"],
            "speaker_id": row["speaker_id"],
            "chapter_id": row["chapter_id"],
            "reference": row["reference"],
            "audio_sha256": row["audio_sha256"],
        }
        digest.update(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    source_hash = sha256(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"source SHA-256 mismatch: {source_hash}")
    rows = pq.read_table(source).to_pylist()
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")

    output_rows = []
    total_seconds = 0.0
    for index, row in enumerate(rows, start=1):
        audio = row["audio"]["bytes"]
        if not audio:
            raise SystemExit(f"missing audio bytes: {row['id']}")
        info = sf.info(io.BytesIO(audio))
        if info.samplerate != 16000 or info.channels != 1:
            raise SystemExit(f"unexpected audio format: {row['id']}")
        duration = info.frames / info.samplerate
        total_seconds += duration
        output_rows.append(
            {
                "sample_id": row["id"],
                "speaker_id": str(row["speaker_id"]),
                "chapter_id": str(row["chapter_id"]),
                "source_dataset": "openslr/librispeech_asr",
                "dataset_split": "test.clean",
                "language_class": "english",
                "duration_seconds": duration,
                "reference": " ".join(row["text"].split()),
                "audio_sha256": hashlib.sha256(audio).hexdigest(),
                "audio": audio,
            }
        )
        if index % 500 == 0:
            print(f"validated {index}/{len(rows)}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), output, compression="zstd")
    metadata = {
        "name": "LibriSpeech test-clean English retention benchmark",
        "source": "openslr/librispeech_asr",
        "source_revision": "main",
        "source_file": "all/test.clean/0000.parquet",
        "license": "CC BY 4.0",
        "rows": len(output_rows),
        "speakers": len({row["speaker_id"] for row in output_rows}),
        "hours": total_seconds / 3600,
        "source_sha256": source_hash,
        "content_fingerprint": content_fingerprint(output_rows),
        "bundle_sha256": sha256(output),
        "bytes": output.stat().st_size,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
