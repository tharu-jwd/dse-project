#!/usr/bin/env python3
"""Create a lightweight canonical Parquet index for an extracted OpenSLR-52 snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/openslr52"))
    parser.add_argument("--output", type=Path, default=Path("data/indexes/openslr52.parquet"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source = args.source if args.source.is_absolute() else project_root / args.source
    output = args.output if args.output.is_absolute() else project_root / args.output
    transcript_path = source / "utt_spk_text.tsv"
    table = csv.read_csv(
        transcript_path,
        read_options=csv.ReadOptions(column_names=["source_record_id", "speaker_id", "text"]),
        parse_options=csv.ParseOptions(delimiter="\t"),
    )
    ids = table["source_record_id"].to_pylist()
    audio_root = source / "asr_sinhala" / "data"
    paths = [(audio_root / value[:2] / f"{value}.flac").resolve() for value in ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise SystemExit(f"{len(missing)} OpenSLR audio files are missing; first paths:\n{preview}")
    canonical = pa.table(
        {
            "audio": pa.array([str(path) for path in paths]),
            "text": table["text"],
            "source_dataset": pa.array(["openslr52"] * len(paths)),
            "source_record_id": table["source_record_id"],
            "speaker_id": table["speaker_id"],
            "upstream_split": pa.array(["unsplit"] * len(paths)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(canonical, output, compression="zstd")
    print(f"Indexed {len(paths)} rows at {output}")


if __name__ == "__main__":
    main()
