#!/usr/bin/env python3
"""Launch E015 phase A from immutable private Kaggle inputs."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

matches = list(Path("/kaggle/input").rglob("run_e015_kaggle.py"))
if len(matches) != 1:
    raise RuntimeError(f"expected one run_e015_kaggle.py, found {matches}")
os.environ["E015_PHASE"] = "phase-a"
runpy.run_path(str(matches[0]), run_name="__main__")
