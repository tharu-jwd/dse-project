#!/usr/bin/env python3
"""Build hash-verified training shards (with embedded audio) for a nested
v4 training subset identified by an exact sample-ID list.

Reads each selected row's audio from the OpenSLR-52 index (the only
upstream source for the v4 train split) using the row's recorded
`source_row_index`, verifies the raw file bytes hash against the frozen
manifest `audio_sha256` for that row, and writes self-contained shards
matching the schema already used by prior nested subsets (E005):
sample_id, speaker_id, language_class, duration_seconds, text (canonical),
audio_sha256, audio (raw bytes). Output is written directly in shards --
no monolithic single-file bundle is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/versions/v4/manifest.parquet"
INDEX = ROOT / "data/indexes/openslr52.parquet"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="output of select_nested_subset.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=1000)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    sample_ids = selection["sample_ids"]
    id_set = set(sample_ids)

    manifest = pq.read_table(
        MANIFEST,
        columns=[
            "sample_id",
            "speaker_id",
            "language_class",
            "duration_seconds",
            "text_canonical",
            "audio_sha256",
            "source_dataset",
            "source_row_index",
            "dataset_split",
        ],
    )
    mask = pc.is_in(manifest.column("sample_id"), value_set=pa.array(id_set))
    selected = manifest.filter(mask)
    if selected.num_rows != len(id_set):
        raise SystemExit(
            f"expected {len(id_set)} manifest rows for the selection, found {selected.num_rows}"
        )
    if any(s != "train" for s in selected.column("dataset_split").to_pylist()):
        raise SystemExit("selection includes a non-train row")
    if any(s != "openslr52" for s in selected.column("source_dataset").to_pylist()):
        raise SystemExit("selection includes a row from an unexpected source dataset")

    rows_by_id = {row["sample_id"]: row for row in selected.to_pylist()}
    ordered_rows = [rows_by_id[sid] for sid in sorted(id_set)]

    index_table = pq.read_table(INDEX, columns=["audio"])
    index_paths = index_table.column("audio").to_pylist()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"refusing to write into non-empty directory: {output}")

    shards = []
    total_rows = 0
    for start in range(0, len(ordered_rows), args.rows_per_shard):
        batch = ordered_rows[start : start + args.rows_per_shard]
        sample_ids_col = []
        speaker_ids_col = []
        language_class_col = []
        duration_col = []
        text_col = []
        audio_sha_col = []
        audio_col = []
        for row in batch:
            index_row = row["source_row_index"]
            audio_path = Path(index_paths[index_row])
            raw = audio_path.read_bytes()
            digest = sha256_bytes(raw)
            if digest != row["audio_sha256"]:
                raise SystemExit(
                    f"audio hash mismatch for {row['sample_id']}: "
                    f"expected {row['audio_sha256']}, got {digest}"
                )
            sample_ids_col.append(row["sample_id"])
            speaker_ids_col.append(row["speaker_id"])
            language_class_col.append(row["language_class"])
            duration_col.append(row["duration_seconds"])
            text_col.append(row["text_canonical"])
            audio_sha_col.append(digest)
            audio_col.append(raw)

        table = pa.table(
            {
                "sample_id": sample_ids_col,
                "speaker_id": speaker_ids_col,
                "language_class": language_class_col,
                "duration_seconds": duration_col,
                "text": text_col,
                "audio_sha256": audio_sha_col,
                "audio": audio_col,
            }
        )
        number = len(shards)
        shard_path = output / f"part-{number:04d}.parquet"
        pq.write_table(table, shard_path, compression="zstd")
        shard_hash = sha256_file(shard_path)
        shards.append(
            {
                "name": shard_path.name,
                "rows": len(batch),
                "bytes": shard_path.stat().st_size,
                "sha256": shard_hash,
            }
        )
        total_rows += len(batch)
        print(f"wrote {shard_path.name}: {len(batch)} rows")

    if total_rows != len(id_set):
        raise SystemExit(f"row count mismatch after sharding: {total_rows} != {len(id_set)}")

    shards_combined_sha256 = hashlib.sha256(
        "".join(sorted(shard["sha256"] for shard in shards)).encode("utf-8")
    ).hexdigest()

    manifest_out = {
        "experiment": selection.get("experiment", "e006"),
        "dataset_fingerprint": sha256_file(MANIFEST),
        "selection_fingerprint": selection["selection_fingerprint"],
        "seed": selection["seed"],
        "rows": total_rows,
        "rows_per_shard": args.rows_per_shard,
        "shards_combined_sha256": shards_combined_sha256,
        "shards": shards,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest_out, indent=2))


if __name__ == "__main__":
    main()
