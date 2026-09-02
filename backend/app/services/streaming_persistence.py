from uuid import UUID

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.transcription import Transcript, TranscriptSegment
from app.models.user import User
from app.services.transcript_service import DuplicateTranscriptTitleError, title_is_taken


def create_live_transcript(owner: User, title: str, transcript_type: str) -> UUID:
    with SessionLocal.begin() as db:
        if title_is_taken(db, owner.user_id, title):
            raise DuplicateTranscriptTitleError(
                f'You already have a transcript titled "{title.strip()}".'
            )

        transcript = Transcript(
            owner_id=owner.user_id,
            media_id=None,
            title=title,
            transcript_type=transcript_type,
            status="DRAFT",
            source="LIVE",
        )
        db.add(transcript)
        db.flush()

        return transcript.transcript_id


def add_final_segment(
    transcript_id: UUID,
    segment_order: int,
    text: str,
    start: float,
    end: float,
    confidence: float = 0.0,
) -> UUID:
    """Persist one final segment immediately - never buffer until session end.

    Returns the new row's id so the caller can hand it to the client (see
    streaming.py's "final" message) - without it the client has no way to
    PATCH /transcripts/{id} for this specific line later, since that
    endpoint identifies segments by their real database id, not the
    in-session `segment_order` the streaming protocol uses.
    """

    with SessionLocal.begin() as db:
        segment = TranscriptSegment(
            transcript_id=transcript_id,
            segment_order=segment_order,
            generated_text=text,
            edited_text=None,
            start_time=start,
            end_time=max(end, start),
            confidence=max(0.0, min(1.0, confidence)),
            word_metadata=None,
        )
        db.add(segment)
        db.flush()

        return segment.segment_id


def delete_last_segment(transcript_id: UUID) -> int | None:
    """Remove the most recently added segment for a live transcript.

    Backs the "delete" voice command: the student speaks a command instead
    of dictated text, so nothing was persisted for that utterance - this
    just un-does the last real segment before it. Returns the removed
    segment's order (so the caller can tell the client what to remove from
    its own local list), or None if the transcript has no segments yet.
    """

    with SessionLocal.begin() as db:
        last_segment = db.scalar(
            select(TranscriptSegment)
            .where(TranscriptSegment.transcript_id == transcript_id)
            .order_by(TranscriptSegment.segment_order.desc())
            .limit(1)
        )

        if last_segment is None:
            return None

        removed_order = last_segment.segment_order
        db.delete(last_segment)

        return removed_order
