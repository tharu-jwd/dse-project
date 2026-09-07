#!/usr/bin/env python3
"""Ingest the Path Nirvana Sinhala TTS dataset as a third external eval source.

Source: /Users/tharupahan/Code/ASR-prior-works/sinhala-tts-dataset (not part of
this repo; a separately cloned, GPL-3-licensed, two-speaker studio TTS corpus).
This script never modifies that source tree. It reads metadata.csv (pipe-
delimited: id|roman_transliteration|sinhala_script|speaker) and the matching
22.05kHz wavs, resamples each clip to this project's 16kHz mono PCM16
convention, and writes a raw-shaped parquet plus a manifest-v2 manifest through
the same pipeline used for every other source (sinhala_asr.data.manifest).

Eval-only: this source is never used for training. See docs/data/dataset.md
and docs/project/plan.md item 2 for the licensing/role note (GPL-3, voices
donated for non-obscene speech generation -- read as permitting private
transcription-accuracy evaluation, not redistribution or resynthesis).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

from sinhala_asr.data.manifest import build_manifest_rows, write_manifest

SOURCE_DATASET = "sinhala-tts-pathnirvana"
TARGET_SAMPLE_RATE = 16000


def _resample_to_16k(path: Path) -> bytes:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != TARGET_SAMPLE_RATE:
        from math import gcd

        g = gcd(sample_rate, TARGET_SAMPLE_RATE)
        up, down = TARGET_SAMPLE_RATE // g, sample_rate // g
        audio = resample_poly(audio, up, down).astype(np.float32)
    clipped = np.clip(audio, -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, clipped, TARGET_SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def _read_metadata(metadata_csv: Path) -> list[tuple[str, str, str]]:
    rows = []
    with metadata_csv.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            record_id, _roman, sinhala, speaker = line.split("|")
            rows.append((record_id, sinhala, speaker))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/Users/tharupahan/Code/ASR-prior-works/sinhala-tts-dataset"),
    )
    parser.add_argument("--raw-output", type=Path, required=True, help="raw-shaped parquet output path")
    parser.add_argument("--manifest-output", type=Path, required=True, help="manifest-v2 parquet output path")
    args = parser.parse_args()

    metadata_csv = args.source_dir / "metadata.csv"
    wav_dir = args.source_dir / "v2.1-audio" / "wavs"
    entries = _read_metadata(metadata_csv)
    print(f"read {len(entries)} metadata rows from {metadata_csv}")

    records = []
    skipped = []
    for record_id, sinhala_text, speaker in entries:
        wav_path = wav_dir / f"{record_id}.wav"
        if not wav_path.exists():
            skipped.append(record_id)
            continue
        audio_bytes = _resample_to_16k(wav_path)
        records.append(
            {
                "audio": {"bytes": audio_bytes},
                "text": sinhala_text,
                "speaker_id": speaker,
                "source_dataset": SOURCE_DATASET,
                "source_record_id": record_id,
                "domain": "studio_tts_read_speech",
            }
        )
    if skipped:
        print(f"WARNING: {len(skipped)} metadata rows had no matching wav file: {skipped[:10]}...")

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, args.raw_output, compression="zstd")
    print(f"wrote raw parquet: {args.raw_output} ({len(records)} rows)")

    rows = build_manifest_rows(split="unsplit", path=args.raw_output)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, args.manifest_output)
    invalid = sum(1 for r in rows if not r["is_valid"])
    total_hours = sum(r["duration_seconds"] or 0.0 for r in rows) / 3600
    print(
        f"wrote manifest: {args.manifest_output} "
        f"({len(rows)} rows, {invalid} invalid, {total_hours:.2f} hours, "
        f"speakers={sorted({r['speaker_id'] for r in rows})})"
    )


if __name__ == "__main__":
    main()
