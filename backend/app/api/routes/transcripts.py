from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import (
    CurrentUserDependency,
    SessionDependency,
)
from app.schemas.transcript import (
    TranscriptListItem,
    TranscriptResponse,
)
from app.services.transcript_service import (
    get_accessible_transcript,
    list_accessible_transcripts,
    serialize_transcript,
    serialize_transcript_list_item,
)


router = APIRouter(
    prefix="/transcripts",
    tags=["Transcripts"],
)


@router.get(
    "",
    response_model=list[TranscriptListItem],
)
def get_transcripts(
    db: SessionDependency,
    current_user: CurrentUserDependency,
) -> list[TranscriptListItem]:
    transcripts = list_accessible_transcripts(
        db=db,
        user=current_user,
    )

    return [
        serialize_transcript_list_item(transcript)
        for transcript in transcripts
    ]


@router.get(
    "/{transcript_id}",
    response_model=TranscriptResponse,
)
def get_transcript(
    transcript_id: UUID,
    db: SessionDependency,
    current_user: CurrentUserDependency,
) -> TranscriptResponse:
    transcript = get_accessible_transcript(
        db=db,
        user=current_user,
        transcript_id=transcript_id,
    )

    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript was not found.",
        )

    return serialize_transcript(transcript)
