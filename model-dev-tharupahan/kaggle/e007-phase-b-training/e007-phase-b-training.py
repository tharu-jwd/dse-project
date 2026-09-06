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
os.environ["E007_SOURCE_COMMIT"] = "b5fe7a21855849508f6a8a9c0c29416031a82304"
os.environ["E007_RESUME_ARCHIVE_SHA256"] = (
    "254ccab6ba327823d5e23d0e19185577175b9ecdd8f2e7f19f9dd67a69bf0508"
)
runpy.run_path(str(one_file("run_e007_kaggle.py")), run_name="__main__")
