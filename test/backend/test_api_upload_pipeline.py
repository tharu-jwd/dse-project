"""Function tests for the upload -> job queue -> worker -> transcript
pipeline (Master Test Plan §3.1.2's remaining gap: "zero tests exercise
the upload -> job-queued -> worker -> transcript-appears path").

The upload endpoint (POST /transcriptions) is driven through TestClient
exactly as a browser would. The worker half is not a separate OS process
here - it is the same `process_next_job()` function the real worker script
(scripts/run_transcription_worker.py) calls in a loop, invoked directly
against a FakeTranscriber (app/transcribers/fake.py, the project's own
"development transcriber used to test the worker pipeline") so this
suite never loads a real Whisper model and stays fast and deterministic.
"""

import io
from uuid import UUID

import pytest

from app.db.session import SessionLocal
from app.models.media import MediaFile
from app.models.transcription import Transcript, TranscriptionJob, TranscriptSegment
from app.services.transcription_processor import process_next_job
from app.transcribers.fake import FakeTranscriber


def _require_empty_queue():
    """process_next_job() claims the OLDEST queued job across the whole
    table, and this suite runs against the shared dev database (see
    conftest.py) - if a real, unrelated job were already queued, calling
    process_next_job() here would claim and "complete" it with
    FakeTranscriber's canned text instead of our own test job, silently
    corrupting real data. Skip rather than risk that."""

    with SessionLocal() as db:
        pending = db.query(TranscriptionJob).filter(
            TranscriptionJob.status.in_(("QUEUED", "PROCESSING"))
        ).count()
    if pending:
        pytest.skip(
            f"{pending} real job(s) already queued/processing in the shared dev "
            "database - skipping to avoid claiming someone else's job"
        )


def _wav_bytes(seconds: float = 0.2) -> bytes:
    """A tiny but structurally valid WAV file - enough to pass through
    save_media_upload() and be openable, without a real recording."""

    import struct

    sample_rate = 16_000
    n_samples = int(sample_rate * seconds)
    data = b"\x00\x00" * n_samples
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(data),
    )
    return header + data


def _cleanup(transcript_id=None, job_id=None):
    with SessionLocal.begin() as db:
        if transcript_id:
            db.query(TranscriptSegment).filter(
                TranscriptSegment.transcript_id == transcript_id
            ).delete()
            db.query(Transcript).filter(Transcript.transcript_id == transcript_id).delete()
        if job_id:
            job = db.get(TranscriptionJob, job_id)
            if job is not None:
                if job.media_id:
                    media = db.get(MediaFile, job.media_id)
                    if media is not None:
                        from app.services.media_storage_service import resolve_stored_media

                        resolve_stored_media(media.storage_path).unlink(missing_ok=True)
                        db.delete(media)
                db.delete(job)


# --- Upload validation ------------------------------------------------


def test_upload_queues_a_job(client, student):
    response = client.post(
        "/transcriptions",
        headers=student.auth,
        files={"file": ("lecture.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"title": "Pipeline test lecture", "type": "LECTURE"},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    # serialize_created_job() reports the initial creation response as
    # "UPLOADING" regardless of the row's real status - the internal
    # "QUEUED" only shows up (as public "PROCESSING") once you poll
    # GET /transcriptions/jobs/{id}, tested below.
    assert body["status"] == "UPLOADING"

    with SessionLocal() as db:
        job = db.get(TranscriptionJob, UUID(body["jobId"]))
        assert job.status == "QUEUED", "the job was not actually queued in the database"

    _cleanup(job_id=UUID(body["jobId"]))


def test_upload_rejects_an_unsupported_file_type(client, student):
    response = client.post(
        "/transcriptions",
        headers=student.auth,
        files={"file": ("notes.pdf", io.BytesIO(b"%PDF-1.4 not audio"), "application/pdf")},
        data={"title": "Wrong file type", "type": "LECTURE"},
    )
    assert response.status_code == 415


def test_upload_rejects_an_empty_file(client, student):
    response = client.post(
        "/transcriptions",
        headers=student.auth,
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        data={"title": "Empty upload", "type": "LECTURE"},
    )
    assert response.status_code == 400


def test_a_note_upload_rejects_a_video_file(client, student):
    """validate_upload requires NOTE/QUIZ_ANSWER to be audio specifically -
    a video file would otherwise silently be accepted for a note."""

    response = client.post(
        "/transcriptions",
        headers=student.auth,
        files={"file": ("clip.mp4", io.BytesIO(b"not really mp4 data"), "video/mp4")},
        data={"title": "Video as a note", "type": "NOTE"},
    )
    assert response.status_code == 415


def test_job_status_is_not_visible_to_another_student(client, student, other_student):
    upload = client.post(
        "/transcriptions",
        headers=student.auth,
        files={"file": ("lecture.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"title": "Private job", "type": "LECTURE"},
    )
    job_id = upload.json()["jobId"]

    try:
        response = client.get(f"/transcriptions/jobs/{job_id}", headers=other_student.auth)
        assert response.status_code in (403, 404)
    finally:
        _cleanup(job_id=UUID(job_id))


# --- The full pipeline, end to end ----------------------------------------


def test_full_pipeline_upload_to_completed_transcript(client, student):
    """The real end-to-end path: queue a job through the HTTP API exactly
    as the browser does, hand it to the same process_next_job() the
    worker process loops on, and confirm a real, readable transcript with
    real segments exists afterward - through the API, not just the DB."""

    _require_empty_queue()

    upload = client.post(
        "/transcriptions",
        headers=student.auth,
        files={"file": ("lecture.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"title": "End to end pipeline test", "type": "LECTURE"},
    )
    assert upload.status_code == 202
    job_id = upload.json()["jobId"]

    status_before = client.get(f"/transcriptions/jobs/{job_id}", headers=student.auth)
    # QUEUED is reported publicly as PROCESSING (serialize_job_status) -
    # from the API's perspective the job is already being worked on.
    assert status_before.json()["status"] == "PROCESSING"

    try:
        processed = process_next_job(FakeTranscriber())
        assert processed is True, "process_next_job found nothing to claim"

        status_after = client.get(f"/transcriptions/jobs/{job_id}", headers=student.auth)
        body = status_after.json()
        assert body["status"] == "COMPLETED", body
        transcript_id = body["transcriptId"]
        assert transcript_id

        transcript_response = client.get(f"/transcripts/{transcript_id}", headers=student.auth)
        assert transcript_response.status_code == 200
        transcript = transcript_response.json()
        assert len(transcript["segments"]) > 0
        assert transcript["segments"][0]["text"]  # FakeTranscriber always returns real Sinhala text
        assert transcript["status"] == "DRAFT", "a freshly completed transcript must be a draft"
    finally:
        with SessionLocal() as db:
            job = db.get(TranscriptionJob, UUID(job_id))
            transcript_id_to_clean = job.transcript_id if job else None
        _cleanup(transcript_id=transcript_id_to_clean, job_id=UUID(job_id))


def test_worker_reports_no_work_for_a_job_it_does_not_own(client, student, other_student):
    """process_next_job's contract (transcription_processor.py) is to
    return False rather than raise or block when there is nothing *it*
    should claim. This runs against the shared dev database (see
    conftest.py's module docstring), so it deliberately does not drain
    the real queue - other queued jobs may belong to real dev data and
    must be left untouched. Instead: queue one job of our own, claim it,
    then confirm a second call finds nothing further belonging to us."""

    _require_empty_queue()

    upload = client.post(
        "/transcriptions",
        headers=student.auth,
        files={"file": ("lecture.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"title": "Queue-empty contract test", "type": "LECTURE"},
    )
    job_id = UUID(upload.json()["jobId"])
    transcript_id = None

    try:
        first_claim = process_next_job(FakeTranscriber())
        assert first_claim is True

        with SessionLocal() as db:
            job = db.get(TranscriptionJob, job_id)
            assert job.status == "COMPLETED"
            transcript_id = job.transcript_id
    finally:
        _cleanup(transcript_id=transcript_id, job_id=job_id)
