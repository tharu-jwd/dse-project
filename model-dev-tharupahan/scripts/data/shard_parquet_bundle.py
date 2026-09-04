#!/usr/bin/env python3
"""Split a self-contained Parquet bundle into ordered, hash-verified shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=1000)
    args = parser.parse_args()
    if args.rows_per_shard <= 0:
        raise SystemExit("rows-per-shard must be positive")
    source = args.input.expanduser().resolve()
    table = pq.read_table(source)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {output}")

    shards = []
    for start in range(0, table.num_rows, args.rows_per_shard):
        number = len(shards)
        shard = output / f"part-{number:04d}.parquet"
        rows = min(args.rows_per_shard, table.num_rows - start)
        pq.write_table(table.slice(start, rows), shard, compression="zstd")
        shards.append(
            {
                "name": shard.name,
                "rows": rows,
                "bytes": shard.stat().st_size,
                "sha256": sha256(shard),
            }
        )
        print(f"wrote {shard.name}: {rows} rows")
    manifest = {
        "source": str(source),
        "source_sha256": sha256(source),
        "rows": table.num_rows,
        "rows_per_shard": args.rows_per_shard,
        "shards": shards,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
