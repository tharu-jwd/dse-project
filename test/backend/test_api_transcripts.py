"""Function tests for the transcript REST API (Master Test Plan §3.1.2).

There is no POST /transcripts - a transcript is only ever created by the
upload pipeline (test_api_upload_pipeline.py) or the live streaming path
(test_streaming_commands_route.py). So CRUD/export/finalize here insert a
transcript directly via the ORM, exactly as either of those real creation
paths would leave one, and then drive the update/export/delete/finalize
routes through TestClient against it.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.db.session import SessionLocal
from app.models.transcription import Transcript, TranscriptSegment


def _make_transcript(owner_id, *, title="Test lecture", transcript_type="LECTURE", segments=None):
    transcript_id = uuid4()
    with SessionLocal.begin() as db:
        db.add(
            Transcript(
                transcript_id=transcript_id,
                owner_id=owner_id,
                title=title,
                transcript_type=transcript_type,
                status="DRAFT",
            )
        )
        for order, (start, end, text) in enumerate(segments or []):
            db.add(
                TranscriptSegment(
                    segment_id=uuid4(),
                    transcript_id=transcript_id,
                    segment_order=order,
                    generated_text=text,
                    edited_text=None,
                    start_time=start,
                    end_time=end,
                    confidence=0.9,
                )
            )
    return transcript_id


def _delete_transcript(transcript_id):
    with SessionLocal.begin() as db:
        db.query(TranscriptSegment).filter(
            TranscriptSegment.transcript_id == transcript_id
        ).delete()
        db.query(Transcript).filter(Transcript.transcript_id == transcript_id).delete()


@pytest.fixture
def transcript(student):
    transcript_id = _make_transcript(
        student.user_id,
        segments=[(0.0, 2.0, "පළමු කොටස"), (2.0, 4.5, "දෙවන කොටස")],
    )
    yield transcript_id
    _delete_transcript(transcript_id)


# --- Read -------------------------------------------------------------


def test_owner_can_read_their_transcript(client, student, transcript):
    response = client.get(f"/transcripts/{transcript}", headers=student.auth)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(transcript)
    assert len(body["segments"]) == 2


def test_transcript_appears_in_owners_list(client, student, transcript):
    response = client.get("/transcripts", headers=student.auth)
    assert response.status_code == 200
    assert str(transcript) in {item["id"] for item in response.json()}


def test_other_student_cannot_read_it(client, other_student, transcript):
    response = client.get(f"/transcripts/{transcript}", headers=other_student.auth)
    assert response.status_code == 404, (
        "a second student was able to read another student's transcript"
    )


def test_teacher_can_read_any_lecture_transcript(client, teacher, transcript):
    """transcript_service.transcript_access_condition grants a teacher
    access to every LECTURE, not just their own - that is a deliberate
    design choice (teachers review student lectures), not a bug, but it
    needs to be pinned down so a future change to that condition is
    caught either way."""

    response = client.get(f"/transcripts/{transcript}", headers=teacher.auth)
    assert response.status_code == 200


def test_teacher_cannot_read_a_students_note(client, teacher, student):
    """The LECTURE carve-out must not extend to NOTE transcripts - those
    are a student's private self-study material."""

    note_id = _make_transcript(student.user_id, transcript_type="NOTE")
    try:
        response = client.get(f"/transcripts/{note_id}", headers=teacher.auth)
        assert response.status_code == 404
    finally:
        _delete_transcript(note_id)


# --- Update -------------------------------------------------------------


def test_owner_can_edit_segment_text(client, student, transcript):
    get_response = client.get(f"/transcripts/{transcript}", headers=student.auth)
    segment_id = get_response.json()["segments"][0]["id"]

    response = client.patch(
        f"/transcripts/{transcript}",
        headers=student.auth,
        json={"segments": [{"id": segment_id, "text": "නිවැරදි කළ පෙළ"}]},
    )

    assert response.status_code == 200
    assert response.json()["segments"][0]["text"] == "නිවැරදි කළ පෙළ"


def test_editing_with_an_unknown_segment_id_is_rejected(client, student, transcript):
    response = client.patch(
        f"/transcripts/{transcript}",
        headers=student.auth,
        json={"segments": [{"id": str(uuid4()), "text": "doesn't belong here"}]},
    )
    assert response.status_code == 400


def test_other_student_cannot_edit_it(client, other_student, transcript):
    response = client.patch(
        f"/transcripts/{transcript}", headers=other_student.auth, json={"title": "hijacked"}
    )
    assert response.status_code == 404


def test_teacher_cannot_edit_a_lecture_they_can_only_read(client, teacher, transcript):
    """The read carve-out (previous test) must not imply write access -
    update_owned_transcript strictly checks ownership, unlike the read path."""

    response = client.patch(
        f"/transcripts/{transcript}", headers=teacher.auth, json={"title": "hijacked"}
    )
    assert response.status_code == 404


def test_finalized_transcript_cannot_be_edited(client, student, transcript):
    finalize = client.post(f"/transcripts/{transcript}/finalize", headers=student.auth)
    assert finalize.status_code == 200
    assert finalize.json()["status"] == "FINALIZED"

    response = client.patch(
        f"/transcripts/{transcript}", headers=student.auth, json={"title": "too late"}
    )
    assert response.status_code == 409


def test_renaming_to_an_already_used_title_is_rejected(client, student):
    """title_is_taken() scopes uniqueness per-owner - creating a second
    transcript and renaming it to collide with the first must fail."""

    first = _make_transcript(student.user_id, title="Unique Title A")
    second = _make_transcript(student.user_id, title="Unique Title B")
    try:
        response = client.patch(
            f"/transcripts/{second}", headers=student.auth, json={"title": "Unique Title A"}
        )
        assert response.status_code == 409
    finally:
        _delete_transcript(first)
        _delete_transcript(second)


# --- Export ---------------------------------------------------------------


def test_owner_can_export_as_txt(client, student, transcript):
    response = client.get(f"/transcripts/{transcript}/export?format=txt", headers=student.auth)
    assert response.status_code == 200
    assert "පළමු කොටස" in response.text
    assert "attachment" in response.headers.get("content-disposition", "")


def test_export_rejects_an_unsupported_format(client, student, transcript):
    response = client.get(f"/transcripts/{transcript}/export?format=pdf", headers=student.auth)
    assert response.status_code == 400


def test_other_student_cannot_export_it(client, other_student, transcript):
    response = client.get(
        f"/transcripts/{transcript}/export?format=txt", headers=other_student.auth
    )
    assert response.status_code == 404


# --- Delete ---------------------------------------------------------------


def test_owner_can_delete_their_transcript(client, student):
    transcript_id = _make_transcript(student.user_id)

    response = client.delete(f"/transcripts/{transcript_id}", headers=student.auth)
    assert response.status_code == 204

    with SessionLocal() as db:
        assert db.get(Transcript, transcript_id) is None


def test_deleting_cascades_its_segments(client, student):
    transcript_id = _make_transcript(
        student.user_id, segments=[(0.0, 1.0, "one"), (1.0, 2.0, "two")]
    )

    client.delete(f"/transcripts/{transcript_id}", headers=student.auth)

    with SessionLocal() as db:
        remaining = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.transcript_id == transcript_id)
            .count()
        )
    assert remaining == 0


def test_other_student_cannot_delete_it(client, other_student, transcript):
    response = client.delete(f"/transcripts/{transcript}", headers=other_student.auth)
    assert response.status_code == 404

    with SessionLocal() as db:
        assert db.get(Transcript, transcript) is not None, (
            "the transcript was deleted despite the request coming from a "
            "non-owner - it must still exist"
        )
