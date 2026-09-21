"""API-level tests for GET /media/{media_id} (Appendix A.6's coverage
gap: media_access_service's ownership/permission logic was covered
indirectly through other tests, but nothing drove the route itself -
missing id, wrong owner, and the teacher-via-LECTURE-transcript carve-out
were all untested at the HTTP layer).

Media rows are created through the real upload endpoint (same helper as
test_api_upload_pipeline.py) rather than inserted directly, so this also
exercises the real on-disk storage path end to end.
"""

import io
import struct
from uuid import UUID, uuid4

from app.db.session import SessionLocal
from app.models.media import MediaFile
from app.models.transcription import Transcript, TranscriptionJob
from app.services.media_storage_service import resolve_stored_media


def _wav_bytes(seconds: float = 0.2) -> bytes:
    sample_rate = 16_000
    n_samples = int(sample_rate * seconds)
    data = b"\x00\x00" * n_samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(data),
    )
    return header + data


def _upload_media(client, owner, media_type="NOTE"):
    response = client.post(
        "/transcriptions",
        headers=owner.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"title": "Media access test clip", "type": media_type},
    )
    assert response.status_code == 202, response.text
    job_id = UUID(response.json()["jobId"])

    with SessionLocal() as db:
        job = db.get(TranscriptionJob, job_id)
        media_id = job.media_id

    return job_id, media_id


def _cleanup(job_id, media_id, transcript_id=None):
    with SessionLocal.begin() as db:
        if transcript_id:
            db.query(Transcript).filter(Transcript.transcript_id == transcript_id).delete()
        job = db.get(TranscriptionJob, job_id)
        if job is not None:
            db.delete(job)
        media = db.get(MediaFile, media_id)
        if media is not None:
            resolve_stored_media(media.storage_path).unlink(missing_ok=True)
            db.delete(media)


def test_get_media_requires_authentication(client):
    response = client.get(f"/media/{uuid4()}")
    assert response.status_code == 401


def test_get_media_rejects_a_nonexistent_id(client, student):
    response = client.get(f"/media/{uuid4()}", headers=student.auth)
    assert response.status_code == 404


def test_owner_can_fetch_their_own_media(client, student):
    job_id, media_id = _upload_media(client, student)
    try:
        response = client.get(f"/media/{media_id}", headers=student.auth)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
    finally:
        _cleanup(job_id, media_id)


def test_a_different_student_cannot_fetch_someone_elses_media(client, student, other_student):
    job_id, media_id = _upload_media(client, student)
    try:
        response = client.get(f"/media/{media_id}", headers=other_student.auth)
        assert response.status_code == 404
    finally:
        _cleanup(job_id, media_id)


def test_a_teacher_cannot_fetch_a_students_private_note(client, student, teacher):
    """NOTE media is private to its owner - the teacher carve-out below
    applies only to LECTURE transcripts, not every recording a student
    happens to own."""

    job_id, media_id = _upload_media(client, student, media_type="NOTE")
    try:
        response = client.get(f"/media/{media_id}", headers=teacher.auth)
        assert response.status_code == 404
    finally:
        _cleanup(job_id, media_id)


def test_a_teacher_can_fetch_media_behind_their_own_lecture_transcript(client, teacher):
    job_id, media_id = _upload_media(client, teacher, media_type="LECTURE")
    with SessionLocal.begin() as db:
        job = db.get(TranscriptionJob, job_id)
        job.status = "COMPLETED"
        transcript = Transcript(
            owner_id=teacher.user_id,
            media_id=media_id,
            title="Media access test lecture",
            transcript_type="LECTURE",
            source="UPLOAD",
        )
        db.add(transcript)
        db.flush()
        transcript_id = transcript.transcript_id

    try:
        response = client.get(f"/media/{media_id}", headers=teacher.auth)
        assert response.status_code == 200
    finally:
        _cleanup(job_id, media_id, transcript_id)
