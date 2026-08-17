import math
import threading
from pathlib import Path
from typing import Any

import whisper

from app.transcribers.base import (
    TranscriptionResult,
    TranscriptionSegmentResult,
    TranscriptionWord,
)


class WhisperTranscriber:
    '''
    Local transcription using OpenAI Whisper small.
    '''

    def __init__(
        self,
        model_name: str = "small",
        language: str = "si",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model is None:
            self._model = whisper.load_model(
                self.model_name,
            )

        return self._model

    def transcribe(
        self,
        media_path: Path,
    ) -> TranscriptionResult:
        if not media_path.is_file():
            raise FileNotFoundError(
                "The uploaded media file does not exist."
            )

        model = self._load_model()

        with self._lock:
            output = model.transcribe(
                str(media_path),
                language=self.language,
                # language = None,
                task="transcribe",
                fp16=False,
                word_timestamps=True,
                verbose=False,
                no_speech_threshold=None
            )

        segments = [
            self._convert_segment(segment)
            for segment in output.get("segments", [])
            if segment.get("text", "").strip()
        ]

        if not segments:
            raise ValueError(
                "Whisper did not detect any speech."
            )

        return TranscriptionResult(
            text=output.get("text", "").strip(),
            segments=segments,
        )

    def _convert_segment(
        self,
        segment: dict[str, Any],
    ) -> TranscriptionSegmentResult:
        segment_confidence = self._confidence_from_log_probability(
            segment.get("avg_logprob"),
        )

        words = [
            TranscriptionWord(
                text=word.get("word", "").strip(),
                confidence=self._word_confidence(word),
            )
            for word in segment.get("words", [])
            if word.get("word", "").strip()
        ]

        return TranscriptionSegmentResult(
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=segment["text"].strip(),
            confidence=segment_confidence,
            words=words,
        )

    @staticmethod
    def _confidence_from_log_probability(
        average_log_probability: float | None,
    ) -> float:
        if average_log_probability is None:
            return 0.0

        return max(
            0.0,
            min(1.0, math.exp(average_log_probability)),
        )

    @staticmethod
    def _word_confidence(
        word: dict[str, Any],
    ) -> float:
        probability = word.get("probability")

        if probability is None:
            return 0.0

        return max(
            0.0,
            min(1.0, float(probability)),
        )
