#!/usr/bin/env python3
"""Build a deterministic, self-contained LibriSpeech train-clean-100 replay set."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
BASE_URL = (
    "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/"
    f"{REVISION}/clean/train.100"
)
PARTS = tuple(f"{number:04d}.parquet" for number in range(14))


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def speaker_balanced_ids(rows: list[dict], count: int, seed: int) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[str(row["speaker_id"])].append(str(row["id"]))
    queues = {
        speaker: deque(sorted(ids, key=lambda item: stable_key(item, seed)))
        for speaker, ids in groups.items()
    }
    speakers = sorted(queues, key=lambda item: stable_key(item, seed))
    selected: list[str] = []
    while speakers and len(selected) < count:
        remaining = []
        for speaker in speakers:
            selected.append(queues[speaker].popleft())
            if queues[speaker]:
                remaining.append(speaker)
            if len(selected) == count:
                break
        speakers = remaining
    if len(selected) != count:
        raise ValueError(f"requested {count} rows but selected {len(selected)}")
    return selected


def audio_bytes(value: object) -> bytes:
    if isinstance(value, dict) and value.get("bytes") is not None:
        return value["bytes"]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ValueError("audio row does not contain embedded bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/content/sinhala-asr-job/source/librispeech-train-clean-100"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/content/sinhala-asr-job/output/e003-english-replay.parquet"),
    )
    parser.add_argument("--rows", type=int, default=1111)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    sources = []
    metadata: list[dict] = []
    for number, name in enumerate(PARTS, start=1):
        path = work / name
        if not path.exists():
            temporary = path.with_suffix(".parquet.partial")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", temporary)
            temporary.replace(path)
        table = pq.read_table(path, columns=["id", "speaker_id"])
        metadata.extend(table.to_pylist())
        sources.append(
            {"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
        print(f"indexed source {number}/{len(PARTS)}", flush=True)

    selected_ids = speaker_balanced_ids(metadata, args.rows, args.seed)
    selected_set = set(selected_ids)
    rows_by_id: dict[str, dict] = {}
    for number, name in enumerate(PARTS, start=1):
        for row in pq.read_table(work / name).to_pylist():
            sample_id = str(row["id"])
            if sample_id not in selected_set:
                continue
            audio = audio_bytes(row["audio"])
            info = sf.info(io.BytesIO(audio))
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(f"unexpected audio format: {sample_id}")
            duration = info.frames / info.samplerate
            if duration > 30:
                raise ValueError(f"selected audio exceeds 30 seconds: {sample_id}")
            rows_by_id[sample_id] = {
                "sample_id": f"librispeech-train-clean-100:{sample_id}",
                "speaker_id": f"librispeech:{row['speaker_id']}",
                "language_class": "english_replay",
                "decoder_language": "en",
                "duration_seconds": duration,
                "text": row["text"],
                "audio_sha256": hashlib.sha256(audio).hexdigest(),
                "audio": audio,
                "source_split": "train-clean-100",
            }
        print(f"extracted source {number}/{len(PARTS)}", flush=True)
    missing = selected_set - rows_by_id.keys()
    if missing:
        raise ValueError(f"selected IDs missing from source: {len(missing)}")
    output_rows = [rows_by_id[sample_id] for sample_id in selected_ids]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), output, compression="zstd")
    record = {
        "dataset": "LibriSpeech ASR corpus",
        "license": "CC BY 4.0",
        "source_repository": "openslr/librispeech_asr",
        "source_revision": REVISION,
        "source_split": "clean/train.100",
        "selection": "deterministic speaker-balanced round robin",
        "seed": args.seed,
        "rows": len(output_rows),
        "speakers": len({row["speaker_id"] for row in output_rows}),
        "hours": sum(row["duration_seconds"] for row in output_rows) / 3600,
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "sources": sources,
    }
    output.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
