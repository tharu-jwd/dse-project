from app.core.config import settings
from app.transcribers.base import Transcriber
from app.transcribers.fake import FakeTranscriber
from app.transcribers.whisper import WhisperTranscriber
from app.transcribers.speak_asr import SpeakAsrTranscriber


def create_transcriber() -> Transcriber:
    backend = settings.transcriber_backend.strip().lower()

    if backend == "fake":
        return FakeTranscriber()

    if backend == "whisper":
        return WhisperTranscriber(
            model_name=settings.whisper_model,
            language=settings.whisper_language,
        )

    if backend == "speak_asr":
        return SpeakAsrTranscriber(
            base_model=settings.whisper_base_model,
            adapter_model=settings.whisper_adapter_model,
        )

    raise RuntimeError(
        f"Unknown transcriber backend: {backend}"
    )
