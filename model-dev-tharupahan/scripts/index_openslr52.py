#!/usr/bin/env python3
"""Create a lightweight canonical Parquet index for an extracted OpenSLR-52 snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_transcripts(path: Path) -> tuple[list[str], list[str], list[str]]:
    record_ids: list[str] = []
    speaker_ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) != 3 or not all(fields):
                raise ValueError(f"invalid OpenSLR TSV row at {path}:{line_number}")
            record_id, speaker_id, text = fields
            record_ids.append(record_id)
            speaker_ids.append(speaker_id)
            texts.append(text.rstrip("\r"))
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("OpenSLR source record IDs are not unique")
    return record_ids, speaker_ids, texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/openslr52"))
    parser.add_argument("--output", type=Path, default=Path("data/indexes/openslr52.parquet"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source = args.source if args.source.is_absolute() else project_root / args.source
    output = args.output if args.output.is_absolute() else project_root / args.output
    transcript_path = source / "utt_spk_text.tsv"
    ids, speakers, texts = read_transcripts(transcript_path)
    audio_root = source / "asr_sinhala" / "data"
    paths = [(audio_root / value[:2] / f"{value}.flac").resolve() for value in ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise SystemExit(f"{len(missing)} OpenSLR audio files are missing; first paths:\n{preview}")
    canonical = pa.table(
        {
            "audio": pa.array([str(path) for path in paths]),
            "text": pa.array(texts),
            "source_dataset": pa.array(["openslr52"] * len(paths)),
            "source_record_id": pa.array(ids),
            "speaker_id": pa.array(speakers),
            "upstream_split": pa.array(["unsplit"] * len(paths)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(canonical, output, compression="zstd")
    print(f"Indexed {len(paths)} rows at {output}")


if __name__ == "__main__":
    main()
