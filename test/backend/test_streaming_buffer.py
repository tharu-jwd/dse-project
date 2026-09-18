"""Unit tests for app.streaming.buffer.StreamingBuffer - the rolling PCM
window used by simulated streaming transcription. Pure numpy, no FastAPI
or database involved, so these run against synthetic PCM only.
"""

import numpy as np
import pytest

from app.streaming.buffer import SAMPLE_RATE, StreamingBuffer


def pcm_chunk(seconds: float, sample_rate: int = SAMPLE_RATE, value: int = 100) -> bytes:
    """A constant-value 16-bit PCM chunk of the given duration."""

    n_samples = int(seconds * sample_rate)
    return np.full(n_samples, value, dtype=np.int16).tobytes()


def test_rejects_overlap_not_smaller_than_max_buffer():
    with pytest.raises(ValueError):
        StreamingBuffer(max_buffer_seconds=1.0, overlap_seconds=1.0)
    with pytest.raises(ValueError):
        StreamingBuffer(max_buffer_seconds=1.0, overlap_seconds=2.0)


def test_starts_empty():
    buffer = StreamingBuffer()
    assert buffer.is_empty
    assert buffer.duration_seconds == 0
    assert buffer.consumed_seconds == 0


def test_append_accumulates_duration():
    buffer = StreamingBuffer()
    buffer.append(pcm_chunk(0.5))
    assert not buffer.is_empty
    assert buffer.duration_seconds == pytest.approx(0.5)

    buffer.append(pcm_chunk(0.25))
    assert buffer.duration_seconds == pytest.approx(0.75)


def test_as_float32_normalizes_into_unit_range():
    buffer = StreamingBuffer()
    buffer.append(np.array([32767, -32768, 0], dtype=np.int16).tobytes())

    floats = buffer.as_float32()
    assert floats == pytest.approx([32767 / 32768.0, -1.0, 0.0])
    assert floats.dtype == np.float32


def test_exceeded_max_buffer_threshold():
    buffer = StreamingBuffer(max_buffer_seconds=1.0, overlap_seconds=0.2)
    buffer.append(pcm_chunk(0.9))
    assert not buffer.exceeded_max_buffer()

    buffer.append(pcm_chunk(0.2))
    assert buffer.exceeded_max_buffer()


def test_exceeded_memory_ceiling_threshold():
    buffer = StreamingBuffer(
        max_buffer_seconds=100.0, overlap_seconds=1.0, memory_ceiling_seconds=2.0
    )
    buffer.append(pcm_chunk(1.5))
    assert not buffer.exceeded_memory_ceiling()

    buffer.append(pcm_chunk(0.5))
    assert buffer.exceeded_memory_ceiling()


def test_finalize_returns_segment_and_clears_buffer():
    buffer = StreamingBuffer()
    buffer.append(pcm_chunk(2.0))

    segment = buffer.finalize("hello world")

    assert segment.text == "hello world"
    assert segment.start == pytest.approx(0.0)
    assert segment.end == pytest.approx(2.0)
    assert buffer.is_empty
    assert buffer.duration_seconds == 0
    assert buffer.consumed_seconds == pytest.approx(2.0)


def test_finalize_segments_stack_on_consumed_seconds():
    buffer = StreamingBuffer()

    buffer.append(pcm_chunk(1.0))
    first = buffer.finalize("first")
    assert first.start == pytest.approx(0.0)
    assert first.end == pytest.approx(1.0)

    buffer.append(pcm_chunk(1.5))
    second = buffer.finalize("second")
    assert second.start == pytest.approx(1.0)
    assert second.end == pytest.approx(2.5)
    assert buffer.consumed_seconds == pytest.approx(2.5)


def test_force_cut_keeps_only_trailing_overlap():
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    # First half at one value, second half at another, so the kept tail is
    # distinguishable from the discarded head.
    buffer.append(pcm_chunk(4.0, value=10))
    buffer.append(pcm_chunk(1.0, value=20))

    buffer.force_cut()

    assert buffer.duration_seconds == pytest.approx(1.0)
    assert buffer.consumed_seconds == pytest.approx(4.0)
    assert np.all(buffer.as_float32() == pytest.approx(20 / 32768.0))


def test_force_cut_on_buffer_shorter_than_overlap_keeps_everything():
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=2.0)
    buffer.append(pcm_chunk(0.5))

    buffer.force_cut()

    # Less audio than the overlap window existed, so nothing is discarded
    # and nothing is counted as permanently consumed.
    assert buffer.duration_seconds == pytest.approx(0.5)
    assert buffer.consumed_seconds == pytest.approx(0.0)


def test_force_cut_with_zero_overlap_clears_buffer():
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=0.0)
    buffer.append(pcm_chunk(3.0))

    buffer.force_cut()

    assert buffer.is_empty
    assert buffer.consumed_seconds == pytest.approx(3.0)


def test_force_cut_then_append_continues_from_kept_tail():
    buffer = StreamingBuffer(max_buffer_seconds=15.0, overlap_seconds=1.0)
    buffer.append(pcm_chunk(5.0))
    buffer.force_cut()
    assert buffer.duration_seconds == pytest.approx(1.0)

    buffer.append(pcm_chunk(2.0))
    assert buffer.duration_seconds == pytest.approx(3.0)

    segment = buffer.finalize("continued")
    assert segment.start == pytest.approx(4.0)
    assert segment.end == pytest.approx(7.0)
