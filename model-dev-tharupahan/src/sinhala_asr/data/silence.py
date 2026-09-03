"""Frame-level leading and trailing silence measurement."""

from __future__ import annotations

from typing import Any

import numpy as np


def boundary_silence(
    samples: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.01,
    frame_ms: float = 20.0,
) -> dict[str, Any]:
    """Measure consecutive quiet frames at both clip boundaries."""
    if samples.ndim == 2:
        samples = np.mean(samples, axis=1)
    samples = np.asarray(samples, dtype=np.float32)
    frame_size = max(1, round(sample_rate * frame_ms / 1000))
    frame_count = (len(samples) + frame_size - 1) // frame_size
    if not frame_count:
        return {
            "leading_silence_seconds": 0.0,
            "trailing_silence_seconds": 0.0,
            "active_audio_seconds": 0.0,
            "boundary_silence_fraction": 0.0,
        }
    padded = np.pad(samples, (0, frame_count * frame_size - len(samples)))
    frames = padded.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    active = rms >= threshold
    if active.any():
        first = int(np.argmax(active))
        last = len(active) - int(np.argmax(active[::-1]))
        leading_frames = first
        trailing_frames = len(active) - last
    else:
        leading_frames = len(active)
        trailing_frames = 0
    duration = len(samples) / sample_rate
    leading = min(duration, leading_frames * frame_size / sample_rate)
    trailing = min(duration - leading, trailing_frames * frame_size / sample_rate)
    boundary = leading + trailing
    return {
        "leading_silence_seconds": leading,
        "trailing_silence_seconds": trailing,
        "active_audio_seconds": max(0.0, duration - boundary),
        "boundary_silence_fraction": boundary / duration if duration else 0.0,
    }
