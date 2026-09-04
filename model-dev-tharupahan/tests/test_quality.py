import numpy as np

from sinhala_asr.data.quality import clipping_severity


def test_clipping_severity_counts_samples_and_longest_run() -> None:
    samples = np.array([0.0, 0.999, 1.0, 0.2, -1.0], dtype=np.float32)
    result = clipping_severity(samples)
    assert result["clipped_samples"] == 3
    assert result["clipped_fraction"] == 0.6
    assert result["longest_clipped_run_samples"] == 2


def test_clipping_severity_uses_loudest_channel() -> None:
    samples = np.array([[0.2, 1.0], [0.3, 0.4]], dtype=np.float32)
    result = clipping_severity(samples)
    assert result["clipped_samples"] == 1
    assert result["longest_clipped_run_samples"] == 1
