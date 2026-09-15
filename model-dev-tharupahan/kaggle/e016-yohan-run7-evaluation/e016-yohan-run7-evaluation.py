#!/usr/bin/env python3
"""Launch the pinned Yohan run7 comparison evaluation."""

from __future__ import annotations

import runpy
from pathlib import Path

matches = list(Path("/kaggle/input").rglob("run_yohan_run7_kaggle.py"))
if len(matches) != 1:
    raise RuntimeError(f"expected one evaluator, found {matches}")
runpy.run_path(str(matches[0]), run_name="__main__")
