import numpy as np
import pytest

from sinhala_asr.data.silence import boundary_silence


def test_boundary_silence_measures_quiet_frames() -> None:
    samples = np.concatenate([np.zeros(20), np.ones(40) * 0.5, np.zeros(40)])
    result = boundary_silence(samples, 100, threshold=0.01, frame_ms=100)
    assert result["leading_silence_seconds"] == 0.2
    assert result["trailing_silence_seconds"] == 0.4
    assert result["active_audio_seconds"] == pytest.approx(0.4)


def test_entirely_silent_clip_is_all_leading_silence() -> None:
    result = boundary_silence(np.zeros(100), 100, frame_ms=100)
    assert result["leading_silence_seconds"] == 1.0
    assert result["active_audio_seconds"] == 0.0
