from pathlib import Path
from typing import Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class TranscriptionWord(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)

    model_config = ConfigDict(extra="forbid")


class TranscriptionSegmentResult(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    confidence: float = Field(ge=0, le=1)
    words: list[TranscriptionWord] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_time_range(self) -> "TranscriptionSegmentResult":
        if self.end < self.start:
            raise ValueError(
                "Segment end time cannot be before its start time."
            )

        return self


class TranscriptionResult(BaseModel):
    text: str
    segments: list[TranscriptionSegmentResult] = Field(
        min_length=1,
    )

    model_config = ConfigDict(extra="forbid")


class Transcriber(Protocol):
    def transcribe(
        self,
        media_path: Path,
    ) -> TranscriptionResult:
        """Transcribe one local media file."""
        ...
