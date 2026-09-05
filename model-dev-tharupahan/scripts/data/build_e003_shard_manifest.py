#!/usr/bin/env python3
"""Compose verified E002 and English-replay shard manifests for E003."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e002-manifest", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    e002 = json.loads(args.e002_manifest.read_text())
    replay = json.loads(args.replay_manifest.read_text())
    if e002["rows"] != 10_000 or replay["rows"] != 1_111:
        raise SystemExit("unexpected E002 or replay row count")
    identity = {
        "experiment": "e003",
        "mixing": "uniform random sampling from concatenated 10,000+1,111 rows",
        "e002_source_sha256": e002["source_sha256"],
        "english_replay_source_sha256": replay["source_sha256"],
        "rows": 11_111,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shards = [
        {**shard, "name": f"e002/{shard['name']}"} for shard in e002["shards"]
    ] + [
        {**shard, "name": f"english-replay/{shard['name']}"}
        for shard in replay["shards"]
    ]
    output = {
        **identity,
        "source_sha256": fingerprint,
        "english_replay_fraction": replay["rows"] / identity["rows"],
        "shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
