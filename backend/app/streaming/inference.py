"""faster-whisper wrapper for streaming inference.

Loaded once at application startup. GPU access is serialised with an
asyncio.Semaphore (not threading.Lock) so a slow transcription only
blocks other streaming sessions, never the event loop itself.
"""

import asyncio

import numpy as np
from faster_whisper import WhisperModel

from app.core.config import settings


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

    async def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""

        async with self._gpu_gate:
            segments_iterator, _info = await asyncio.to_thread(
                self._model.transcribe,
                audio,
                language=self._language,
                task="transcribe",
                vad_filter=False,
                without_timestamps=True,
            )
            segments = await asyncio.to_thread(list, segments_iterator)

        return " ".join(segment.text.strip() for segment in segments).strip()


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
