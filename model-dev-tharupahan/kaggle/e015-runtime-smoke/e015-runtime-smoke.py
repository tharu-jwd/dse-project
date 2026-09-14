#!/usr/bin/env python3
"""Launch the exact E015 mixed-language checkpoint/resume smoke."""

from __future__ import annotations

import runpy
from pathlib import Path

matches = list(Path("/kaggle/input").rglob("run_e015_kaggle_smoke.py"))
if len(matches) != 1:
    raise RuntimeError(f"expected one run_e015_kaggle_smoke.py, found {matches}")
runpy.run_path(str(matches[0]), run_name="__main__")
