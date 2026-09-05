#!/usr/bin/env python3
"""Package the completed adapter with a transport hash before session teardown."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

output = Path("/content/sinhala-asr-job/output")
adapter = output / "final-adapter"
required = {"adapter_config.json", "adapter_model.safetensors"}
missing = required - {path.name for path in adapter.iterdir()}
if missing:
    raise SystemExit(f"final adapter files missing: {sorted(missing)}")
archive = output / "final-adapter.tar.gz"
partial = output / "final-adapter.tar.gz.partial"
with tarfile.open(partial, "w:gz") as bundle:
    bundle.add(adapter, arcname="final-adapter")
partial.replace(archive)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
metadata = {"name": archive.name, "bytes": archive.stat().st_size, "sha256": digest}
(output / "final-artifact.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(metadata, indent=2))
