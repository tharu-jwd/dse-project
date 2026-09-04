#!/usr/bin/env python3
"""Prepare phase B from a locally verified, re-uploaded phase-A checkpoint."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

root = Path("/content/sinhala-asr-job")
output = root / "output"
archive_output = root / "completed-smoke-phase-a"
resume_archive = Path("/content/resume-checkpoint.tar.gz")
resume_dir = root / "input" / "resume"
if archive_output.exists():
    raise SystemExit(f"refusing to overwrite {archive_output}")
if not resume_archive.is_file():
    raise SystemExit(f"missing {resume_archive}")
output.rename(archive_output)
output.mkdir()
if resume_dir.exists() and any(resume_dir.iterdir()):
    raise SystemExit(f"refusing to overwrite {resume_dir}")
resume_dir.mkdir(parents=True, exist_ok=True)
with tarfile.open(resume_archive, "r:gz") as bundle:
    bundle.extractall(resume_dir, filter="data")
checkpoints = [path for path in resume_dir.iterdir() if path.is_dir()]
if len(checkpoints) != 1:
    raise SystemExit("resume archive must contain one checkpoint")
checkpoint = checkpoints[0]
step = int((checkpoint / "COMPLETE").read_text().strip())
trainer_step = int(
    json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
)
if step != trainer_step:
    raise SystemExit(f"resume step mismatch: {step}/{trainer_step}")
print(json.dumps({"resume_checkpoint": str(checkpoint), "step": step}))
