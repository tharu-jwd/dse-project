"""Regression test: the shared VAD must survive concurrent streaming sessions.

Found by load testing (test/load): the streaming route calls `vad.analyze` via
`asyncio.to_thread`, so concurrent sessions run it on several threads at once
against ONE shared, stateful Silero model. Unserialised, that corrupted the
model's internal state and raised `RuntimeError: select(): index 1 out of range
for tensor of size [1, 64]` mid-session.
"""

import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from app.streaming.vad import VoiceActivityDetector

SAMPLES = Path(__file__).resolve().parents[2] / "storage" / "voice_samples"


def _clips() -> list[np.ndarray]:
    clips = []
    for path in sorted(SAMPLES.glob("*.wav"))[:6]:
        with wave.open(str(path), "rb") as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        clips.append(pcm.astype(np.float32) / 32768.0)
    return clips


def test_concurrent_analyze_matches_sequential_results():
    clips = _clips()
    if not clips:
        pytest.skip("no voice sample clips available")

    detector = VoiceActivityDetector()
    expected = [detector.analyze(clip) for clip in clips]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(detector.analyze, clips[i % len(clips)]) for i in range(80)]
        results = [future.result() for future in futures]  # re-raises any RuntimeError

    for i, result in enumerate(results):
        want = expected[i % len(clips)]
        assert result.has_speech == want.has_speech
        assert result.trailing_silence_seconds == pytest.approx(
            want.trailing_silence_seconds, abs=1e-6
        )
