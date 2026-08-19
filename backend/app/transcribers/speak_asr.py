import threading
from pathlib import Path
from typing import Any

import whisper
from peft import PeftModel
from transformers import (
    AutoProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)

from app.core.config import settings

from app.transcribers.base import (
    TranscriptionResult,
    TranscriptionSegmentResult,
)


DEFAULT_BASE_MODEL = settings.whisper_base_model
DEFAULT_ADAPTER_MODEL = settings.whisper_adapter_model

class SpeakAsrTranscriber:
    '''
    Sinhala transcription using Whisper base model and SPEAK-ASR LoRA.
    '''

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_model: str = DEFAULT_ADAPTER_MODEL,
    ) -> None:
        self.base_model = base_model
        self.adapter_model = adapter_model
        self._pipeline: Any | None = None
        self._lock = threading.Lock()

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        processor = AutoProcessor.from_pretrained(
            self.base_model,
        )

        whisper_model = WhisperForConditionalGeneration.from_pretrained(
                        self.base_model,
                    )

        adapted_model = PeftModel.from_pretrained(whisper_model, self.adapter_model,)

        merged_model = adapted_model.merge_and_unload()
        merged_model.eval()

        self._pipeline = pipeline(
            task = "automatic-speech-recognition",
            model = merged_model,
            tokenizer = processor.tokenizer,
            feature_extractor = processor.feature_extractor,
            chunk_length_s = 30,
            stride_length_s = 5, 
        )

        return self._pipeline

    def transcribe(self, media_path: Path) -> TranscriptionResult:
        if not media_path.is_file():
            raise FileNotFoundError("The uploaded media file does not exist")

        audio = whisper.load_audio(str(media_path))
        if audio.shape[0] < 1600:   # less that 0.1s
            raise ValueError("The recording is too short.")

        with self._lock:
            asr_pipeline = self._load_pipeline()
            forced_decoder_ids = (
                asr_pipeline.tokenizer.get_decoder_prompt_ids(
                    language="si",
                    task="transcribe",
                )
            )

            output = asr_pipeline(
                {
                    "array": audio,
                    "sampling_rate": 16000,
                },
                return_timestamps=True,
                generate_kwargs={
                    "forced_decoder_ids": forced_decoder_ids,
                    "max_new_tokens": 440,
                },
            )

        text = output.get("text", "").strip()
        segments = self._convert_segments(output)

        # database contract requires atleast one segment
        if not segments:
            duration = len(audio)/16000
            segments = [
                TranscriptionSegmentResult(
                    start=0.0,
                    end=max(duration, 0.01),
                    text=text,
                    confidence=1.0,
                    words=[],
                )
            ]

        return TranscriptionResult(text=text, segments=segments)


    @staticmethod
    def _convert_segments(output: dict[str, Any]) -> list[TranscriptionSegmentResult]:
        segments: list[TranscriptionSegmentResult] = []

        for chunk in output.get("chunks", []):
            timestamp = chunk.get("timestamp")

            if not timestamp or timestamp[0] is None:
                continue

            start = float(timestamp[0])
            end = float(timestamp[1] if timestamp[1] is not None else start)
            chunk_text = chunk.get("text", "").strip()

            if not chunk_text:
                continue

            segments.append(
                TranscriptionSegmentResult(
                    start=start,
                    end=max(start,end),
                    text=chunk_text,
                    confidence=1.0,  # [fix: need to be calculated]
                    words=[],
                )
            )

        return segments
