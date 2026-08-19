import threading
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)

from app.transcribers.base import (
    TranscriptionResult,
    TranscriptionSegmentResult,
)


class WhisperTranscriber:
    """Local transcription using a complete Hugging Face Whisper checkpoint."""

    def __init__(self, model_name: str, language: str = "si") -> None:
        self.model_name = model_name
        self.language = language
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        model_path = Path(self.model_name)
        if model_path.is_absolute() and not model_path.is_dir():
            raise FileNotFoundError(
                f"The configured Whisper model directory does not exist: {model_path}"
            )

        processor = AutoProcessor.from_pretrained(self.model_name)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        # Checkpoints saved by Transformers 4.x may store a single EOS token
        # as a one-item list. Transformers 5.x beam search requires an integer.
        eos_token_id = model.generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            if len(eos_token_id) != 1:
                raise ValueError(
                    "The Whisper checkpoint defines multiple EOS token IDs, "
                    "which this transcription pipeline does not support."
                )
            model.generation_config.eos_token_id = eos_token_id[0]

        model.eval()

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model.to("cuda")

        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if use_cuda else -1,
        )
        return self._pipeline

    def transcribe(self, media_path: Path) -> TranscriptionResult:
        if not media_path.is_file():
            raise FileNotFoundError("The uploaded media file does not exist.")

        asr_pipeline = self._load_pipeline()
        generate_kwargs = {"task": "transcribe"}
        if self.language:
            generate_kwargs["language"] = self.language

        with self._lock:
            output = asr_pipeline(
                str(media_path),
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
            )

        text = output.get("text", "").strip()
        segments = self._convert_segments(output)
        if not text or not segments:
            raise ValueError("Whisper did not detect any speech.")

        return TranscriptionResult(text=text, segments=segments)

    @staticmethod
    def _convert_segments(
        output: dict[str, Any],
    ) -> list[TranscriptionSegmentResult]:
        segments: list[TranscriptionSegmentResult] = []

        for chunk in output.get("chunks", []):
            timestamp = chunk.get("timestamp")
            chunk_text = chunk.get("text", "").strip()
            if not timestamp or timestamp[0] is None or not chunk_text:
                continue

            start = float(timestamp[0])
            end = float(timestamp[1] if timestamp[1] is not None else start)
            segments.append(
                TranscriptionSegmentResult(
                    start=start,
                    end=max(start, end),
                    text=chunk_text,
                    # This pipeline output does not include calibrated confidence.
                    confidence=0.0,
                    words=[],
                )
            )

        return segments
