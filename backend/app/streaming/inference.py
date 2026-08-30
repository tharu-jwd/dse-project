"""faster-whisper wrapper for streaming inference.

Loaded once at application startup. GPU access is serialised with an
asyncio.Semaphore (not threading.Lock) so a slow transcription only
blocks other streaming sessions, never the event loop itself.
"""

import asyncio
import inspect
from dataclasses import dataclass
from typing import Literal

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import settings
from app.streaming.commands import HOTWORDS


TranscriptionMode = Literal["dictation", "command"]


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    avg_logprob: float


def build_transcribe_kwargs(
    mode: TranscriptionMode,
    *,
    hotwords_supported: bool,
    hotwords_enabled: bool = True,
) -> dict:
    """Decoding-bias kwargs for `WhisperModel.transcribe`.

    Pure and model-free so it can be unit tested without loading Whisper.
    Dictation mode always returns `{}` - hotword biasing must never pull
    normal speech toward the command vocabulary.
    """

    if mode != "command" or not hotwords_enabled:
        return {}

    return {"hotwords": HOTWORDS} if hotwords_supported else {"initial_prompt": HOTWORDS}


class StreamingTranscriber:
    def __init__(
        self,
        model_path: str,
        compute_type: str = "int8",
        language: str = "si",
    ) -> None:
        self._model = WhisperModel(
            model_path,
            device="cuda" if _cuda_available() else "cpu",
            compute_type=compute_type,
        )

        try:
            # torch.cuda.is_available() can be True while the CTranslate2/
            # cuBLAS runtime still fails to load (e.g. mismatched CUDA
            # versions) - that only surfaces on the first real encode call.
            list(self._model.transcribe(np.zeros(1600, dtype=np.float32))[0])
        except RuntimeError:
            self._model = WhisperModel(
                model_path,
                device="cpu",
                compute_type=compute_type,
            )
        self._language = language
        self._gpu_gate = asyncio.Semaphore(1)
        self._hotwords_supported = "hotwords" in inspect.signature(
            self._model.transcribe
        ).parameters

    async def transcribe(
        self,
        audio: np.ndarray,
        *,
        mode: TranscriptionMode = "dictation",
    ) -> TranscriptionResult:
        if audio.size == 0:
            return TranscriptionResult(text="", avg_logprob=0.0)

        bias_kwargs = build_transcribe_kwargs(
            mode,
            hotwords_supported=self._hotwords_supported,
            hotwords_enabled=settings.voice_command_hotwords_enabled,
        )

        async with self._gpu_gate:
            segments_iterator, _info = await asyncio.to_thread(
                self._model.transcribe,
                audio,
                language=self._language,
                task="transcribe",
                vad_filter=False,
                without_timestamps=True,
                **bias_kwargs,
            )
            segments = await asyncio.to_thread(list, segments_iterator)

        text = " ".join(segment.text.strip() for segment in segments).strip()
        avg_logprob = (
            sum(segment.avg_logprob for segment in segments) / len(segments)
            if segments
            else 0.0
        )

        return TranscriptionResult(text=text, avg_logprob=avg_logprob)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


_transcriber: StreamingTranscriber | None = None


def get_streaming_transcriber() -> StreamingTranscriber:
    global _transcriber

    if _transcriber is None:
        _transcriber = StreamingTranscriber(
            model_path=settings.streaming_model_ct2_path,
            compute_type=settings.streaming_compute_type,
            language=settings.whisper_language,
        )

    return _transcriber
