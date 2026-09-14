#!/usr/bin/env python3
"""Package only the new E015 controls and v6 validation audio for Kaggle.

The 12 GB E007 private Kaggle dataset already contains every v5 training row,
so E015 reuses those immutable audio shards and uploads only a sample-ID
allow-list. This avoids duplicating the corpus. V6 validation and English
teacher replay are packaged self-contained because they are comparatively
small and must resolve without local filesystem paths in Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.data.manifest import _audio_bytes, _first_present


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_robin(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    groups = {
        str(speaker): group.sort_values("sample_id").to_dict("records")
        for speaker, group in frame.groupby("speaker_id", sort=True)
    }
    selected: list[dict] = []
    depth = 0
    while len(selected) < count:
        added = False
        for speaker in sorted(groups):
            rows = groups[speaker]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError(f"only {len(selected)} rows available for monitor")
        depth += 1
    return pd.DataFrame(selected)


def package_audio(source_manifest: pd.DataFrame, output: Path) -> None:
    source_cache: dict[str, list[dict]] = {}
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("speaker_id", pa.string()),
            ("language_class", pa.string()),
            ("duration_seconds", pa.float64()),
            ("text", pa.string()),
            ("audio_sha256", pa.string()),
            ("audio", pa.binary()),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output, schema, compression="zstd")
    try:
        batch: list[dict] = []
        for row in source_manifest.sort_values("sample_id").to_dict("records"):
            source_path = str(row["source_path"])
            if source_path not in source_cache:
                source_cache[source_path] = pq.read_table(source_path).to_pylist()
            source_row = source_cache[source_path][int(row["source_row_index"])]
            audio = _audio_bytes(
                _first_present(source_row, "audio", "audio_path", "file"),
                Path(source_path).parent,
            )
            batch.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "speaker_id": str(row["speaker_id"]),
                    "language_class": str(row["language_class"]),
                    "duration_seconds": float(row["duration_seconds"]),
                    "text": str(row["text_canonical"]),
                    "audio_sha256": str(row["audio_sha256"]),
                    "audio": audio,
                }
            )
            if len(batch) == 250:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("reports/dataset-audit/openslr-train-v2/manifest.parquet"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("reports/dataset-audit/v6-text-disjoint/validation-manifest.parquet"),
    )
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        default=Path("reports/dataset-audit/teacher-replay-english/manifest.parquet"),
    )
    parser.add_argument(
        "--replay-audio",
        type=Path,
        default=Path("data/raw/teacher-replay-english/data.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/kaggle/e015-input-dataset"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {args.output_dir}")

    train = pd.read_parquet(args.train)
    validation = pd.read_parquet(args.validation)
    replay = pd.read_parquet(args.replay_manifest)
    if len(train) != 165_055 or train["sample_id"].nunique() != len(train):
        raise ValueError("unexpected v5 training identity")
    if len(validation) != 6_493 or validation["sample_id"].nunique() != len(validation):
        raise ValueError("unexpected v6 validation identity")
    if len(replay) != 1_111 or replay["sample_id"].nunique() != len(replay):
        raise ValueError("unexpected teacher replay identity")

    ids_path = args.output_dir / "v5-train-allowlist.parquet"
    pq.write_table(
        pa.Table.from_pandas(
            train[
                ["sample_id", "speaker_id", "text_canonical", "audio_sha256"]
            ].sort_values("sample_id"),
            preserve_index=False,
        ),
        ids_path,
        compression="zstd",
    )

    validation_audio = args.output_dir / "v6-validation-audio.parquet"
    package_audio(validation, validation_audio)
    validation_manifest = validation.sort_values("sample_id").copy()
    validation_manifest["source_path"] = validation_audio.name
    validation_manifest["source_row_index"] = range(len(validation_manifest))
    validation_manifest["decoder_language"] = "si"
    validation_manifest_path = args.output_dir / "v6-validation-manifest.parquet"
    pq.write_table(
        pa.Table.from_pandas(validation_manifest, preserve_index=False),
        validation_manifest_path,
        compression="zstd",
    )

    sinhala_monitor = round_robin(
        validation[validation["language_class"] == "sinhala_only"], 192
    )
    latin_monitor = round_robin(
        validation[validation["language_class"] == "latin_only"], 8
    )
    monitor_ids = set(sinhala_monitor["sample_id"]) | set(latin_monitor["sample_id"])
    monitor = validation_manifest[validation_manifest["sample_id"].isin(monitor_ids)]
    if len(monitor) != 200 or monitor["speaker_id"].nunique() != 52:
        raise ValueError("monitor must contain 200 rows and all 52 validation speakers")
    monitor_path = args.output_dir / "v6-validation-monitor-200.parquet"
    pq.write_table(
        pa.Table.from_pandas(monitor.sort_values("sample_id"), preserve_index=False),
        monitor_path,
        compression="zstd",
    )

    replay_manifest = replay.sort_values("sample_id").copy()
    replay_manifest["source_path"] = args.replay_audio.name
    replay_manifest["decoder_language"] = "en"
    replay_manifest_path = args.output_dir / "teacher-replay-manifest.parquet"
    pq.write_table(
        pa.Table.from_pandas(replay_manifest, preserve_index=False),
        replay_manifest_path,
        compression="zstd",
    )
    replay_audio_path = args.output_dir / args.replay_audio.name
    replay_audio_path.hardlink_to(args.replay_audio.resolve())

    assets = [
        ids_path,
        validation_audio,
        validation_manifest_path,
        monitor_path,
        replay_manifest_path,
        replay_audio_path,
    ]
    metadata = {
        "experiment": "e015",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "monitor_rows": len(monitor),
        "monitor_speakers": int(monitor["speaker_id"].nunique()),
        "monitor_language_classes": {
            str(key): int(value)
            for key, value in monitor["language_class"].value_counts().items()
        },
        "teacher_replay_unique_rows": len(replay),
        "source_hashes": {
            "train_manifest": sha256(args.train),
            "validation_manifest": sha256(args.validation),
            "teacher_replay_manifest": sha256(args.replay_manifest),
            "teacher_replay_audio": sha256(args.replay_audio),
        },
        "assets": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in assets
        ],
    }
    (args.output_dir / "asset-index.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Sinhala ASR E015 Private Training Controls and V6 Validation",
                "id": "tharupahan/sinhala-asr-e015-inputs",
                "licenses": [{"name": "other"}],
                "description": (
                    "Private E015 v5 training allow-list, teacher replay, and "
                    "transcript-disjoint v6 validation. External test excluded."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
