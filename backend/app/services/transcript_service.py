from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.models.transcription import Transcript, TranscriptSegment
from app.models.user import User
from app.schemas.transcript import (
    TranscriptListItem,
    TranscriptResponse,
    TranscriptSegmentResponse,
)


def transcript_access_condition(user: User):
    own_transcript = Transcript.owner_id == user.user_id

    if user.role == "TEACHER":
        return or_(
            own_transcript,
            Transcript.transcript_type == "LECTURE",
        )

    return own_transcript


def list_accessible_transcripts(
    db: Session,
    user: User,
) -> list[Transcript]:
    statement = (
        select(Transcript)
        .where(transcript_access_condition(user))
        .order_by(Transcript.created_at.desc())
    )

    return list(db.scalars(statement).all())


def get_accessible_transcript(
    db: Session,
    user: User,
    transcript_id: UUID,
) -> Transcript | None:
    statement = (
        select(Transcript)
        .options(
            joinedload(Transcript.media_file),
            selectinload(Transcript.segments),
        )
        .where(
            Transcript.transcript_id == transcript_id,
            transcript_access_condition(user),
        )
    )

    return db.scalar(statement)


def serialize_segment(
    segment: TranscriptSegment,
) -> TranscriptSegmentResponse:
    text = (
        segment.edited_text
        if segment.edited_text is not None
        else segment.generated_text
    )

    words = (
        segment.word_metadata
        if isinstance(segment.word_metadata, list)
        else []
    )

    return TranscriptSegmentResponse(
        id=segment.segment_id,
        start_time=segment.start_time,
        end_time=segment.end_time,
        confidence=segment.confidence,
        text=text,
        words=words,
    )


def serialize_transcript_list_item(
    transcript: Transcript,
) -> TranscriptListItem:
    return TranscriptListItem(
        id=transcript.transcript_id,
        owner_id=transcript.owner_id,
        title=transcript.title,
        type=transcript.transcript_type,
        status=transcript.status,
        date=transcript.created_at,
    )


def serialize_transcript(
    transcript: Transcript,
) -> TranscriptResponse:
    media_type = (
        transcript.media_file.media_type
        if transcript.media_file is not None
        else ""
    )

    return TranscriptResponse(
        id=transcript.transcript_id,
        owner_id=transcript.owner_id,
        title=transcript.title,
        type=transcript.transcript_type,
        status=transcript.status,
        date=transcript.created_at,
        media_url="",
        media_type=media_type,
        segments=[
            serialize_segment(segment)
            for segment in transcript.segments
        ],
    )
