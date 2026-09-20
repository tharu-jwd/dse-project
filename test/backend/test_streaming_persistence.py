"""Direct tests for app.services.streaming_persistence (Appendix A.6's
coverage note: this file was 33% covered - reached only through a live
WebSocket session in practice, via test/failover/, but never called
directly). No socket or model needed: these are plain functions that
take a user/id and write rows, so they're driven the same way as any
other service-layer test in this suite.
"""

import uuid

import pytest

from app.db.session import SessionLocal
from app.models.transcription import Transcript, TranscriptSegment
from app.services.streaming_persistence import (
    add_final_segment,
    create_live_transcript,
    delete_last_segment,
)


@pytest.fixture
def live_transcript(student):
    transcript_id = create_live_transcript(student, "Live persistence test", "NOTE")
    yield transcript_id
    with SessionLocal.begin() as db:
        db.query(TranscriptSegment).filter(
            TranscriptSegment.transcript_id == transcript_id
        ).delete()
        db.query(Transcript).filter(Transcript.transcript_id == transcript_id).delete()


def test_create_live_transcript_persists_a_draft_row(student, live_transcript):
    with SessionLocal() as db:
        transcript = db.get(Transcript, live_transcript)
        assert transcript is not None
        assert transcript.owner_id == student.user_id
        assert transcript.status == "DRAFT"
        assert transcript.source == "LIVE"
        assert transcript.media_id is None


def test_create_live_transcript_with_a_duplicate_title_replaces_the_earlier_one(student):
    """create_live_transcript calls the same replace_duplicate_title() the
    non-live create path uses: a same-titled DRAFT is deleted, not
    renamed, so a retry after an error reuses the title cleanly."""

    first_id = create_live_transcript(student, "Duplicate live title", "NOTE")
    second_id = create_live_transcript(student, "Duplicate live title", "NOTE")
    try:
        with SessionLocal() as db:
            assert db.get(Transcript, first_id) is None
            second = db.get(Transcript, second_id)
            assert second is not None
            assert second.title == "Duplicate live title"
    finally:
        with SessionLocal.begin() as db:
            for transcript_id in (first_id, second_id):
                db.query(TranscriptSegment).filter(
                    TranscriptSegment.transcript_id == transcript_id
                ).delete()
                db.query(Transcript).filter(Transcript.transcript_id == transcript_id).delete()


def test_add_final_segment_persists_immediately_and_returns_its_id(live_transcript):
    segment_id = add_final_segment(live_transcript, 0, "hello world", start=0.0, end=1.5, confidence=0.9)

    with SessionLocal() as db:
        segment = db.get(TranscriptSegment, segment_id)
        assert segment is not None
        assert segment.transcript_id == live_transcript
        assert segment.segment_order == 0
        assert segment.generated_text == "hello world"
        assert segment.edited_text is None
        assert segment.start_time == 0.0
        assert segment.end_time == 1.5
        assert segment.confidence == 0.9


def test_add_final_segment_clamps_confidence_into_zero_one(live_transcript):
    too_high = add_final_segment(live_transcript, 0, "a", 0.0, 1.0, confidence=5.0)
    too_low = add_final_segment(live_transcript, 1, "b", 0.0, 1.0, confidence=-5.0)

    with SessionLocal() as db:
        assert db.get(TranscriptSegment, too_high).confidence == 1.0
        assert db.get(TranscriptSegment, too_low).confidence == 0.0


def test_add_final_segment_never_lets_end_precede_start(live_transcript):
    """A malformed VAD window (end < start) must not corrupt the stored
    range - clamped to start, same invariant the CHECK constraint enforces
    at the database level."""

    segment_id = add_final_segment(live_transcript, 0, "clip", start=2.0, end=1.0)

    with SessionLocal() as db:
        segment = db.get(TranscriptSegment, segment_id)
        assert segment.end_time >= segment.start_time


def test_delete_last_segment_removes_the_most_recent_one(live_transcript):
    add_final_segment(live_transcript, 0, "first", 0.0, 1.0)
    second_id = add_final_segment(live_transcript, 1, "second", 1.0, 2.0)

    removed_order = delete_last_segment(live_transcript)

    assert removed_order == 1
    with SessionLocal() as db:
        assert db.get(TranscriptSegment, second_id) is None
        remaining = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.transcript_id == live_transcript)
            .all()
        )
        assert len(remaining) == 1
        assert remaining[0].generated_text == "first"


def test_delete_last_segment_on_an_empty_transcript_returns_none(live_transcript):
    assert delete_last_segment(live_transcript) is None


def test_delete_last_segment_on_a_transcript_with_no_segments_table_rows_for_an_unknown_id():
    assert delete_last_segment(uuid.uuid4()) is None
