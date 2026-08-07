from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    field_validator,
)


TranscriptType = Literal[
    "LECTURE",
    "NOTE",
    "QUIZ_ANSWER",
]

TranscriptStatus = Literal[
    "DRAFT",
    "FINALIZED",
    "PROCESSING",
    "FAILED",
]


class WordMetadataResponse(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)

    model_config = ConfigDict(
        extra="ignore",
    )


class TranscriptSegmentResponse(BaseModel):
    id: UUID
    start_time: float = Field(
        ge=0,
        serialization_alias="startTime",
    )
    end_time: float = Field(
        ge=0,
        serialization_alias="endTime",
    )
    confidence: float = Field(ge=0, le=1)
    text: str
    words: list[WordMetadataResponse] = Field(
        default_factory=list,
    )

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "TranscriptSegmentResponse":
        if self.end_time < self.start_time:
            raise ValueError(
                "Segment end time cannot be before its start time."
            )

        return self


class TranscriptListItem(BaseModel):
    id: UUID
    owner_id: UUID = Field(
        serialization_alias="ownerId",
    )
    title: str
    type: TranscriptType
    status: TranscriptStatus
    date: datetime

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )


class TranscriptResponse(TranscriptListItem):
    media_url: str = Field(
        default="",
        serialization_alias="mediaUrl",
    )
    media_type: str = Field(
        default="",
        serialization_alias="mediaType",
    )
    segments: list[TranscriptSegmentResponse] = Field(
        default_factory=list,
    )


class TranscriptSegmentUpdate(BaseModel):
    id: UUID
    text: str = Field(max_length=10_000)

    model_config = ConfigDict(
        extra="ignore",
    )


class TranscriptUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )
    segments: list[TranscriptSegmentUpdate] | None = None

    model_config = ConfigDict(
        extra="ignore",
    )

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized_title = value.strip()

        if not normalized_title:
            raise ValueError("Transcript title cannot be empty.")

        return normalized_title

    @model_validator(mode="after")
    def validate_update(self) -> "TranscriptUpdate":
        if self.title is None and self.segments is None:
            raise ValueError("At least one transcript field must be provided")

        if self.segments is not None:
            segment_ids = [
                segment.id
                for segment in self.segments
            ]

            if len(segment_ids) != len(set(segment_ids)):
                raise ValueError("A transcript segment cannot apper more than once")

        return self

