#!/usr/bin/env python3
"""Fingerprint every downloaded source file without modifying raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sinhala_asr.data.sources import inventory_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("configs/data/sources.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/sources/inventory.json"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    registry = args.registry if args.registry.is_absolute() else project_root / args.registry
    output = args.output if args.output.is_absolute() else project_root / args.output
    result = inventory_registry(registry, project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for source in result["sources"]:
        inventory = source["inventory"]
        print(f"{source['id']}: {inventory['file_count']} files, {inventory['total_bytes']} bytes")
    print(f"Inventory: {output}")


if __name__ == "__main__":
    main()
