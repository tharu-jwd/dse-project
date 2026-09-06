#!/usr/bin/env python3
"""Launch phase A of E007 from its frozen private Kaggle inputs."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

INPUTS = Path("/kaggle/input")


def one_file(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0]


os.environ["E007_PHASE"] = "phase-a"
os.environ["E007_SOURCE_COMMIT"] = "b5fe7a21855849508f6a8a9c0c29416031a82304"
runpy.run_path(str(one_file("run_e007_kaggle.py")), run_name="__main__")
