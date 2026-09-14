from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/data/build_text_disjoint_evaluation.py"
SPEC = importlib.util.spec_from_file_location("build_text_disjoint_evaluation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_text_values_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="null or empty"):
        MODULE.text_values(pd.DataFrame({"text_metric": ["valid", " "]}))


def test_assert_disjoint_checks_transcripts_as_well_as_audio() -> None:
    left = pd.DataFrame(
        {
            "sample_id": ["a"],
            "audio_sha256": ["audio-a"],
            "audio_pcm_sha256": ["pcm-a"],
            "text_metric": ["shared prompt"],
        }
    )
    right = pd.DataFrame(
        {
            "sample_id": ["b"],
            "audio_sha256": ["audio-b"],
            "audio_pcm_sha256": ["pcm-b"],
            "text_metric": ["shared prompt"],
        }
    )
    with pytest.raises(ValueError, match="overlap in text_metric"):
        MODULE.assert_disjoint(left, right, ("train", "validation"))
