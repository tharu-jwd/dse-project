"""Stuck-job recovery (Task Gap README: "a transcription job interrupted
while in PROCESSING" was previously undefined behaviour - a crashed
worker left its in-flight job PROCESSING forever, since claim_next_job()
only ever looked at QUEUED rows). Fixed in transcription_processor.py by
also reclaiming a PROCESSING job whose started_at is older than
settings.transcription_stuck_job_timeout_minutes.

Runs against the shared dev database like test_db_concurrency.py, so it
uses the same empty_job_queue guard - claim_next_job() takes the oldest
eligible row across the WHOLE table.
"""

import io
import struct
from datetime import datetime, timedelta, timezone
from uuid import UUID

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.media import MediaFile
from app.models.transcription import TranscriptionJob
from app.services.media_storage_service import resolve_stored_media
from app.services.transcription_processor import claim_next_job, process_next_job
from app.transcribers.fake import FakeTranscriber


def _wav_bytes() -> bytes:
    sample_rate = 16_000
    data = b"\x00\x00" * int(sample_rate * 0.2)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(data),
    )
    return header + data


def _upload(client, owner, title: str) -> UUID:
    # Distinct titles per call matter here, not just for clarity:
    # create_transcription_job() cancels any still-pending job with the
    # SAME title for the same user before creating a new one - reusing a
    # title across two uploads in one test would silently delete the
    # first job.
    response = client.post(
        "/transcriptions",
        headers=owner.auth,
        files={"file": ("clip.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"title": title, "type": "NOTE"},
    )
    assert response.status_code == 202, response.text
    return UUID(response.json()["jobId"])


def _cleanup(job_id):
    with SessionLocal.begin() as db:
        job = db.get(TranscriptionJob, job_id)
        if job is None:
            return
        media = db.get(MediaFile, job.media_id) if job.media_id else None
        if media is not None:
            try:
                resolve_stored_media(media.storage_path).unlink()
            except FileNotFoundError:
                pass  # this test deliberately deletes it before cleanup
            db.delete(media)
        db.delete(job)


def test_a_job_stuck_in_processing_past_the_timeout_is_reclaimed(client, student, empty_job_queue):
    job_id = _upload(client, student, "Stuck job test - reclaim")
    try:
        stale = datetime.now(timezone.utc) - timedelta(
            minutes=settings.transcription_stuck_job_timeout_minutes + 1
        )
        with SessionLocal.begin() as db:
            job = db.get(TranscriptionJob, job_id)
            job.status = "PROCESSING"
            job.started_at = stale

        reclaimed = claim_next_job()
        assert reclaimed == job_id, "a stuck PROCESSING job past the timeout must be reclaimed"

        with SessionLocal() as db:
            job = db.get(TranscriptionJob, job_id)
            assert job.status == "PROCESSING"
    finally:
        _cleanup(job_id)


def test_a_job_still_within_the_timeout_is_left_alone(client, student, empty_job_queue):
    job_id = _upload(client, student, "Stuck job test - within timeout")
    try:
        with SessionLocal.begin() as db:
            job = db.get(TranscriptionJob, job_id)
            job.status = "PROCESSING"
            job.started_at = datetime.now(timezone.utc)

        assert claim_next_job() is None, (
            "a PROCESSING job well within its timeout must not be reclaimed - "
            "that would let two workers run the same job at once"
        )
    finally:
        _cleanup(job_id)


def test_a_job_whose_media_was_deleted_fails_cleanly_and_does_not_block_the_queue(
    client, student, empty_job_queue
):
    missing_job_id = _upload(client, student, "Stuck job test - missing media")
    next_job_id = _upload(client, student, "Stuck job test - next in queue")
    try:
        with SessionLocal.begin() as db:
            job = db.get(TranscriptionJob, missing_job_id)
            media = db.get(MediaFile, job.media_id)
            resolve_stored_media(media.storage_path).unlink()

        assert process_next_job(FakeTranscriber()) is True

        with SessionLocal() as db:
            failed = db.get(TranscriptionJob, missing_job_id)
            assert failed.status == "FAILED"
            assert failed.error_message

        # The queue must not be stuck behind the failure - the next job
        # is still claimable and completes normally.
        assert process_next_job(FakeTranscriber()) is True

        with SessionLocal() as db:
            completed = db.get(TranscriptionJob, next_job_id)
            assert completed.status == "COMPLETED"
    finally:
        _cleanup(missing_job_id)
        _cleanup(next_job_id)
