from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.models.transcription import Transcript, TranscriptSegment
from app.models.user import User
from app.schemas.transcript import (
    TranscriptListItem,
    TranscriptResponse,
    TranscriptSegmentResponse,
    TranscriptUpdate,
)


class TranscriptNotFoundError(ValueError):
    pass


class TranscriptFinalizedError(ValueError):
    pass


class TranscriptSegmentMismatchError(ValueError):
    pass


class DuplicateTranscriptTitleError(ValueError):
    pass


def title_is_taken(
    db: Session,
    owner_id: UUID,
    title: str,
    exclude_transcript_id: UUID | None = None,
) -> bool:
    """Case-insensitive check: does this owner already have a transcript
    with this title? Titles are only required to be unique per-owner, not
    globally - two students can both call a note "Chapter 1"."""

    statement = select(Transcript.transcript_id).where(
        Transcript.owner_id == owner_id,
        func.lower(Transcript.title) == title.strip().lower(),
    )

    if exclude_transcript_id is not None:
        statement = statement.where(Transcript.transcript_id != exclude_transcript_id)

    return db.scalar(statement) is not None


def transcript_access_condition(user: User):
    own_transcript = Transcript.owner_id == user.user_id

    if user.role == "TEACHER":
        return or_(
            own_transcript,
            Transcript.transcript_type == "LECTURE",
        )

    return own_transcript


def transcript_load_options():
    return (
        joinedload(Transcript.media_file),
        selectinload(Transcript.segments),
    )


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
        .options(*transcript_load_options())
        .where(
            Transcript.transcript_id == transcript_id,
            transcript_access_condition(user),
        )
    )

    return db.scalar(statement)


def get_owned_transcript(
    db: Session,
    user: User,
    transcript_id: UUID,
) -> Transcript | None:
    statement = (
        select(Transcript)
        .options(*transcript_load_options())
        .where(
            Transcript.transcript_id == transcript_id,
            Transcript.owner_id == user.user_id,
        )
    )

    return db.scalar(statement)


def update_owned_transcript(
    db: Session,
    user: User,
    transcript_id: UUID,
    update_data: TranscriptUpdate,
) -> Transcript:
    transcript = get_owned_transcript(
        db=db,
        user=user,
        transcript_id=transcript_id,
    )

    if transcript is None:
        raise TranscriptNotFoundError(
            "Transcript was not found."
        )

    if transcript.status == "FINALIZED":
        raise TranscriptFinalizedError(
            "A finalized transcript cannot be edited."
        )

    segment_updates = update_data.segments

    if segment_updates is not None:
        existing_segments = {
            segment.segment_id: segment
            for segment in transcript.segments
        }

        unknown_segment_ids = {
            segment_update.id
            for segment_update in segment_updates
            if segment_update.id not in existing_segments
        }

        if unknown_segment_ids:
            raise TranscriptSegmentMismatchError(
                "One or more segments do not belong to this transcript."
            )

    if update_data.title is not None and update_data.title.strip().lower() != transcript.title.strip().lower():
        if title_is_taken(db, transcript.owner_id, update_data.title, exclude_transcript_id=transcript.transcript_id):
            raise DuplicateTranscriptTitleError(
                f'You already have a transcript titled "{update_data.title.strip()}".'
            )
        transcript.title = update_data.title

    if segment_updates is not None:
        for segment_update in segment_updates:
            segment = existing_segments[segment_update.id]
            segment.edited_text = segment_update.text

    transcript.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(transcript)

    return transcript


def finalize_owned_transcript(
    db: Session,
    user: User,
    transcript_id: UUID,
) -> Transcript:
    transcript = get_owned_transcript(
        db=db,
        user=user,
        transcript_id=transcript_id,
    )

    if transcript is None:
        raise TranscriptNotFoundError(
            "Transcript was not found."
        )

    if transcript.status != "FINALIZED":
        transcript.status = "FINALIZED"
        transcript.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(transcript)

    return transcript


def delete_owned_transcript(
    db: Session,
    user: User,
    transcript_id: UUID,
) -> None:
    transcript = get_owned_transcript(
        db=db,
        user=user,
        transcript_id=transcript_id,
    )

    if transcript is None:
        raise TranscriptNotFoundError(
            "Transcript was not found."
        )

    db.delete(transcript)
    db.commit()


def serialize_segment(
    segment: TranscriptSegment,
) -> TranscriptSegmentResponse:
    segment_text = (
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
        text=segment_text,
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
        media_url=(
            f"/media/{transcript.media_id}"
            if transcript.media_id is not None
            else ""
        ),
        media_type=media_type,
        segments=[
            serialize_segment(segment)
            for segment in transcript.segments
        ],
    )
