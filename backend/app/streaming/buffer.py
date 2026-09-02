"""Rolling-window buffer for simulated streaming transcription.

Deliberately free of FastAPI and database imports so it can be unit
tested with plain synthetic PCM arrays, independent of the network/
persistence layers.
"""

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class FinalizedSegment:
    text: str
    start: float
    end: float


class StreamingBuffer:
    """Accumulates PCM audio for one connection and tracks cut/finalize state.

    Whisper needs full context, so the whole buffer is re-transcribed on
    every window tick rather than only the newest chunk. Two kinds of cuts
    exist:
      - a VAD-triggered finalize, which clears the buffer entirely and
        yields a FinalizedSegment to persist.
      - a forced cut (hard cap reached with no pause), which keeps the
        trailing `overlap_seconds` of audio so a word split by the cut
        survives into the next window, but does not finalize anything.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        max_buffer_seconds: float = 15.0,
        overlap_seconds: float = 1.0,
        memory_ceiling_seconds: float = 60.0,
    ) -> None:
        if max_buffer_seconds <= overlap_seconds:
            raise ValueError(
                "max_buffer_seconds must be greater than overlap_seconds."
            )

        self.sample_rate = sample_rate
        self.max_buffer_seconds = max_buffer_seconds
        self.overlap_seconds = overlap_seconds
        self.memory_ceiling_seconds = memory_ceiling_seconds

        self._pcm = np.zeros(0, dtype=np.int16)
        self._consumed_seconds = 0.0

    def append(self, chunk: bytes) -> None:
        """Append a raw 16-bit little-endian mono PCM chunk."""

        samples = np.frombuffer(chunk, dtype=np.int16)
        self._pcm = np.concatenate([self._pcm, samples])

    @property
    def duration_seconds(self) -> float:
        return len(self._pcm) / self.sample_rate

    @property
    def is_empty(self) -> bool:
        return len(self._pcm) == 0

    @property
    def consumed_seconds(self) -> float:
        """Total audio time permanently removed from the front of the buffer."""

        return self._consumed_seconds

    def as_float32(self) -> np.ndarray:
        """Current buffer contents, normalised to [-1.0, 1.0] for Whisper."""

        return self._pcm.astype(np.float32) / 32768.0

    def exceeded_max_buffer(self) -> bool:
        return self.duration_seconds >= self.max_buffer_seconds

    def exceeded_memory_ceiling(self) -> bool:
        return self.duration_seconds >= self.memory_ceiling_seconds

    def finalize(self, text: str) -> FinalizedSegment:
        """VAD detected a pause: emit a final segment and clear the buffer."""

        segment = FinalizedSegment(
            text=text,
            start=self._consumed_seconds,
            end=self._consumed_seconds + self.duration_seconds,
        )
        self._consumed_seconds += self.duration_seconds
        self._pcm = np.zeros(0, dtype=np.int16)

        return segment

    def force_cut(self) -> None:
        """Hard cap reached with no pause: keep only the trailing overlap."""

        overlap_samples = int(self.overlap_seconds * self.sample_rate)
        kept_seconds = min(self.overlap_seconds, self.duration_seconds)

        self._consumed_seconds += self.duration_seconds - kept_seconds
        self._pcm = (
            self._pcm[-overlap_samples:]
            if overlap_samples > 0
            else np.zeros(0, dtype=np.int16)
        )
