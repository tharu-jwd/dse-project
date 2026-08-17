from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


PublicJobStatus = Literal[
    "UPLOADING",
    "PROCESSING",
    "COMPLETED",
    "FAILED",
]


class TranscriptionJobCreatedResponse(BaseModel):
    job_id: UUID = Field(serialization_alias="jobId")
    status: PublicJobStatus

    model_config = ConfigDict(populate_by_name=True)


class TranscriptionJobStatusResponse(BaseModel):
    job_id: UUID = Field(serialization_alias="jobId")
    status: PublicJobStatus
    progress: int = Field(ge=0, le=100)
    transcript_id: UUID | None = Field(
        default=None,
        serialization_alias="transcriptId",
    )
    message: str | None = None

    model_config = ConfigDict(populate_by_name=True)
