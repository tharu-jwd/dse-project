#!/usr/bin/env python3
"""Launch phase B of E007, resuming from the verified phase A checkpoint."""

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


os.environ["E007_PHASE"] = "phase-b"
os.environ["E007_SOURCE_COMMIT"] = "642e813db8181059ae6013774eda8fca7084d7c2"
runpy.run_path(str(one_file("run_e007_kaggle.py")), run_name="__main__")
