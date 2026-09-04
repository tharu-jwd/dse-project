#!/usr/bin/env python3
"""Print remote job status and recent logs without mutating the job."""

from __future__ import annotations

import json
import os
from pathlib import Path

OUTPUT = Path("/content/sinhala-asr-job/output")
process_file = OUTPUT / "process.json"
status_file = OUTPUT / "status.json"
log_file = OUTPUT / "remote-stdout.log"

process = json.loads(process_file.read_text()) if process_file.exists() else {}
pid = process.get("pid")
alive = False
if isinstance(pid, int):
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        pass
status = json.loads(status_file.read_text()) if status_file.exists() else None
tail = ""
if log_file.exists():
    tail = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
print(json.dumps({"pid": pid, "alive": alive, "status": status}, indent=2))
print("RECENT LOG OUTPUT")
print(tail)
