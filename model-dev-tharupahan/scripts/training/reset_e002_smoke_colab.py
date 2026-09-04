#!/usr/bin/env python3
"""Archive the rejected scheduler-mismatch smoke and reset only run state."""

from __future__ import annotations

from pathlib import Path

root = Path("/content/sinhala-asr-job")
rejected = root / "rejected-schedule-mismatch"
if rejected.exists():
    raise SystemExit(f"refusing to overwrite {rejected}")
rejected.mkdir()
for source, name in (
    (root / "completed-smoke-phase-a", "phase-a"),
    (root / "output", "phase-b"),
    (root / "input" / "resume", "resume-input"),
):
    if source.exists():
        source.rename(rejected / name)
(root / "output").mkdir()
print("rejected smoke archived; training inputs preserved")
