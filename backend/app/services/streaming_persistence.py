from uuid import UUID

from app.db.session import SessionLocal
from app.models.transcription import Transcript, TranscriptSegment
from app.models.user import User


def create_live_transcript(owner: User, title: str, transcript_type: str) -> UUID:
    with SessionLocal.begin() as db:
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
) -> None:
    """Persist one final segment immediately - never buffer until session end."""

    with SessionLocal.begin() as db:
        db.add(
            TranscriptSegment(
                transcript_id=transcript_id,
                segment_order=segment_order,
                generated_text=text,
                edited_text=None,
                start_time=start,
                end_time=max(end, start),
                confidence=max(0.0, min(1.0, confidence)),
                word_metadata=None,
            )
        )
