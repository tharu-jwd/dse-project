from app.core.config import settings
from app.transcribers.base import Transcriber
from app.transcribers.fake import FakeTranscriber


def create_transcriber() -> Transcriber:
    backend = settings.transcriber_backend.strip().lower()

    if backend == "fake":
        return FakeTranscriber()

    raise RuntimeError(
        f"Unknown transcriber backend: {backend}"
    )
