#!/usr/bin/env python3
"""Export a self-contained, audio-bearing evaluation split for Colab."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-queue", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pq.read_table(args.manifest).to_pylist()
    queue = pq.read_table(args.audio_queue, columns=["sample_id", "audio"]).to_pylist()
    audio_by_id = {row["sample_id"]: row["audio"] for row in queue}
    selected = [row for row in manifest if row["dataset_split"] == args.split]
    if not selected:
        raise SystemExit(f"split has no rows: {args.split}")

    output_rows = []
    for row in selected:
        audio = audio_by_id.get(row["sample_id"])
        if audio is None:
            raise SystemExit(f"missing embedded audio: {row['sample_id']}")
        if hashlib.sha256(audio).hexdigest() != row["audio_sha256"]:
            raise SystemExit(f"encoded-audio hash mismatch: {row['sample_id']}")
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "language_class": row["language_class"],
                "duration_seconds": row["duration_seconds"],
                "reference": row["text_canonical"],
                "audio_sha256": row["audio_sha256"],
                "audio": audio,
            }
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), output, compression="zstd")
    metadata = {
        "source_manifest": str(args.manifest),
        "dataset_fingerprint": "d232747fbf019f06a6449404d3d0251e8f4547ed02471482c07d85014c81abdb",
        "split": args.split,
        "rows": len(output_rows),
        "bundle_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
