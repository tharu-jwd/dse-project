"""Immutable-source inventory and content fingerprinting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_source(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".cache" in path.parts or path.name.endswith(".part"):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    fingerprint_material = "\n".join(
        f"{item['path']}\t{item['bytes']}\t{item['sha256']}" for item in files
    )
    return {
        "exists": root.is_dir(),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "content_fingerprint": hashlib.sha256(fingerprint_material.encode()).hexdigest(),
        "files": files,
    }


def inventory_registry(registry_path: Path, project_root: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    sources = []
    for source in registry["sources"]:
        item = dict(source)
        item["inventory"] = inventory_source(project_root / source["local_path"])
        sources.append(item)
    return {"schema_version": registry["schema_version"], "sources": sources}
