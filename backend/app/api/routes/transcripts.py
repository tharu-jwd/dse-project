from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.api.dependencies import (
    CurrentUserDependency,
    SessionDependency,
)
from app.schemas.transcript import (
    TranscriptListItem,
    TranscriptResponse,
    TranscriptUpdate,
)
from app.services.transcript_service import (
    TranscriptFinalizedError,
    TranscriptNotFoundError,
    TranscriptSegmentMismatchError,
    delete_owned_transcript,
    finalize_owned_transcript,
    get_accessible_transcript,
    list_accessible_transcripts,
    serialize_transcript,
    serialize_transcript_list_item,
    update_owned_transcript,
)


def render_transcript_text(transcript: TranscriptResponse) -> str:
    lines = [transcript.title, ""]
    lines.extend(segment.text for segment in transcript.segments)
    return "\n".join(lines)


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


@router.patch(
    "/{transcript_id}",
    response_model=TranscriptResponse,
)
def update_transcript(
    transcript_id: UUID,
    update_data: TranscriptUpdate,
    db: SessionDependency,
    current_user: CurrentUserDependency,
) -> TranscriptResponse:
    try:
        transcript = update_owned_transcript(
            db=db,
            user=current_user,
            transcript_id=transcript_id,
            update_data=update_data,
        )
    except TranscriptNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except TranscriptFinalizedError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except TranscriptSegmentMismatchError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return serialize_transcript(transcript)


@router.get(
    "/{transcript_id}/export",
    response_class=PlainTextResponse,
)
def export_transcript(
    transcript_id: UUID,
    db: SessionDependency,
    current_user: CurrentUserDependency,
    format: str = "txt",
) -> PlainTextResponse:
    if format != "txt":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only the 'txt' export format is supported.",
        )

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

    response = serialize_transcript(transcript)

    return PlainTextResponse(
        content=render_transcript_text(response),
        headers={
            "Content-Disposition": f'attachment; filename="{response.title}.txt"',
        },
    )


@router.delete(
    "/{transcript_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_transcript(
    transcript_id: UUID,
    db: SessionDependency,
    current_user: CurrentUserDependency,
) -> None:
    try:
        delete_owned_transcript(
            db=db,
            user=current_user,
            transcript_id=transcript_id,
        )
    except TranscriptNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error


@router.post(
    "/{transcript_id}/finalize",
    response_model=TranscriptResponse,
)
def finalize_transcript(
    transcript_id: UUID,
    db: SessionDependency,
    current_user: CurrentUserDependency,
) -> TranscriptResponse:
    try:
        transcript = finalize_owned_transcript(
            db=db,
            user=current_user,
            transcript_id=transcript_id,
        )
    except TranscriptNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return serialize_transcript(transcript)
