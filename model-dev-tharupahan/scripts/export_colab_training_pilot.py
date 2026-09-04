#!/usr/bin/env python3
"""Export a deterministic, self-contained v4 training pilot for Colab."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.data.manifest import _audio_bytes, _first_present
from sinhala_asr.training.pilot import select_pilot_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--latin-rows", type=int, default=75)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    manifest = pq.read_table(args.manifest).to_pylist()
    selected = select_pilot_rows(
        manifest, total=args.rows, latin_rows=args.latin_rows, seed=args.seed
    )
    source_cache: dict[str, list[dict]] = {}
    output_rows = []
    for index, row in enumerate(selected, start=1):
        source_path = Path(row["source_path"])
        key = str(source_path)
        if key not in source_cache:
            source_cache[key] = pq.read_table(source_path).to_pylist()
        source_row = source_cache[key][int(row["source_row_index"])]
        audio = _audio_bytes(
            _first_present(source_row, "audio", "audio_path", "file"),
            source_path.parent,
        )
        if hashlib.sha256(audio).hexdigest() != row["audio_sha256"]:
            raise SystemExit(f"audio hash mismatch: {row['sample_id']}")
        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "language_class": row["language_class"],
                "duration_seconds": row["duration_seconds"],
                "text": row["text_canonical"],
                "audio_sha256": row["audio_sha256"],
                "audio": audio,
            }
        )
        if index % 250 == 0:
            print(f"packed {index}/{len(selected)}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), output, compression="zstd")
    metadata = {
        "dataset_fingerprint": "d232747fbf019f06a6449404d3d0251e8f4547ed02471482c07d85014c81abdb",
        "split": "train",
        "selection": "deterministic speaker-balanced within language class",
        "seed": args.seed,
        "rows": len(output_rows),
        "sinhala_rows": sum(r["language_class"] != "latin_only" for r in output_rows),
        "latin_rows": sum(r["language_class"] == "latin_only" for r in output_rows),
        "speakers": len({r["speaker_id"] for r in output_rows}),
        "hours": sum(r["duration_seconds"] for r in output_rows) / 3600,
        "bundle_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
