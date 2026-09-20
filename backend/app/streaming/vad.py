"""Silero VAD wrapper: loaded once, reused for every streaming session."""

import threading
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from app.streaming.buffer import SAMPLE_RATE


@dataclass(frozen=True)
class VadResult:
    has_speech: bool
    trailing_silence_seconds: float


class VoiceActivityDetector:
    def __init__(self) -> None:
        self._model = load_silero_vad()
        # The model carries mutable internal state and is not thread-safe, yet
        # every streaming session shares this one instance and calls analyze()
        # on a worker thread (asyncio.to_thread). Unserialised, concurrent
        # sessions corrupt that state - RuntimeError at best, a hard crash of
        # the whole server process at worst. VAD is cheap, so a lock costs little.
        self._lock = threading.Lock()

    def analyze(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> VadResult:
        if audio.size == 0:
            return VadResult(has_speech=False, trailing_silence_seconds=0.0)

        tensor = torch.from_numpy(audio)
        with self._lock:
            timestamps = get_speech_timestamps(
                tensor,
                self._model,
                sampling_rate=sample_rate,
                return_seconds=True,
            )

        buffer_duration = audio.size / sample_rate

        if not timestamps:
            return VadResult(
                has_speech=False,
                trailing_silence_seconds=buffer_duration,
            )

        last_speech_end = timestamps[-1]["end"]
        trailing_silence = max(0.0, buffer_duration - last_speech_end)

        return VadResult(has_speech=True, trailing_silence_seconds=trailing_silence)


_detector: VoiceActivityDetector | None = None


def get_vad() -> VoiceActivityDetector:
    global _detector

    if _detector is None:
        _detector = VoiceActivityDetector()

    return _detector
