"""Audio quality measurements that do not modify source waveforms."""

from __future__ import annotations

import numpy as np


def clipping_severity(
    samples: np.ndarray, *, threshold: float = 0.999
) -> dict[str, float | int]:
    """Measure the amount and longest run of samples at the clipping threshold."""
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        values = np.max(np.abs(values), axis=1)
    else:
        values = np.abs(values.reshape(-1))
    clipped = values >= threshold
    count = int(clipped.sum())
    longest = 0
    current = 0
    for is_clipped in clipped:
        current = current + 1 if is_clipped else 0
        longest = max(longest, current)
    total = int(clipped.size)
    return {
        "clipped_samples": count,
        "clipped_fraction": count / total if total else 0.0,
        "longest_clipped_run_samples": longest,
    }
