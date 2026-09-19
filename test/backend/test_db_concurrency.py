"""Concurrency and atomicity tests for the transcription job queue
(Master Test Plan §3.1.1's "Not yet done" list).

Two properties the queue depends on and that were previously unverified:

  - `claim_next_job()` uses `SELECT ... FOR UPDATE SKIP LOCKED`, so several
    workers can share one queue without ever handing the same job to two of
    them. It had only ever run with a single worker.
  - `complete_job()` writes a Transcript, its segments, and the job's status
    in one transaction. If any part fails, none of it may persist.

All of these run against the shared dev database, and `claim_next_job()`
takes the oldest QUEUED row across the WHOLE table - so every test here
uses the `empty_job_queue` fixture, which skips rather than risk claiming a
real job. Jobs created here are owned by the fixture student, so the
fixture's teardown removes them (and their media files) afterwards.
"""

import io
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import pytest
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.transcription import Transcript, TranscriptionJob
from app.services.transcription_processor import claim_next_job, process_next_job
from app.transcribers.base import TranscriptionResult, TranscriptionSegmentResult


def _wav_bytes() -> bytes:
    sample_rate = 16_000
    data = b"\x00\x00" * int(sample_rate * 0.2)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(data),
    )
    return header + data


def _queue_job(client, user, title: str) -> UUID:
    response = client.post(
        "/transcriptions",
        headers=user.auth,
        files={"file": ("lecture.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
        data={"title": title, "type": "LECTURE"},
    )
    assert response.status_code == 202, response.text
    return UUID(response.json()["jobId"])


def _claim_statement():
    """The same query claim_next_job() runs - kept identical on purpose, so
    the lock-behaviour tests below exercise the real mechanism rather than
    a lookalike."""

    return (
        select(TranscriptionJob)
        .where(TranscriptionJob.status == "QUEUED")
        .order_by(TranscriptionJob.created_at.asc())
        .with_for_update(skip_locked=True)
        .limit(1)
    )


# --- SKIP LOCKED: several workers, one queue --------------------------------


def test_a_second_worker_gets_a_different_job_while_the_first_holds_a_lock(
    client, student, empty_job_queue
):
    """Deterministic version of the race: worker A locks the oldest job and
    has not committed yet; worker B runs the same claim query and must be
    handed the *other* job immediately, not the locked one and not a wait."""

    first = _queue_job(client, student, "Concurrency job one")
    second = _queue_job(client, student, "Concurrency job two")

    with SessionLocal() as worker_a, SessionLocal() as worker_b:
        claimed_by_a = worker_a.scalar(_claim_statement())
        assert claimed_by_a is not None
        assert claimed_by_a.job_id == first, "the oldest job should be claimed first"

        claimed_by_b = worker_b.scalar(_claim_statement())
        assert claimed_by_b is not None
        assert claimed_by_b.job_id == second, (
            "worker B was handed the job worker A already holds - SKIP LOCKED "
            "is not protecting the queue"
        )

        worker_a.rollback()
        worker_b.rollback()


def test_a_locked_job_is_skipped_not_waited_on(client, student, empty_job_queue):
    """With only one job and it locked, a second worker must get nothing back
    promptly. If SKIP LOCKED were missing the query would block until the
    first worker finished - stalling every idle worker behind one busy one."""

    _queue_job(client, student, "Only job")

    with SessionLocal() as worker_a, SessionLocal() as worker_b:
        assert worker_a.scalar(_claim_statement()) is not None

        result = {}

        def second_worker():
            result["job"] = worker_b.scalar(_claim_statement())

        thread = threading.Thread(target=second_worker)
        thread.start()
        thread.join(timeout=5)

        assert not thread.is_alive(), (
            "the second worker blocked on a locked row instead of skipping it"
        )
        assert result["job"] is None

        worker_a.rollback()
        worker_b.rollback()


def test_concurrent_claim_next_job_never_hands_out_the_same_job(
    client, student, empty_job_queue
):
    """The real function, called from two threads at once, over several jobs."""

    job_ids = {_queue_job(client, student, f"Race job {n}") for n in range(4)}

    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        claimed = []
        while (job_id := claim_next_job()) is not None:
            claimed.append(job_id)
        return claimed

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [f.result() for f in [pool.submit(worker), pool.submit(worker)]]

    all_claimed = results[0] + results[1]
    assert len(all_claimed) == len(set(all_claimed)), (
        f"the same job was claimed twice: {all_claimed}"
    )
    assert set(all_claimed) == job_ids, "some queued jobs were never claimed"


# --- Atomicity and failure handling -----------------------------------------


class _RaisingTranscriber:
    def transcribe(self, media_path):
        raise RuntimeError("boom: internal detail that must not reach the user")


class _DbInvalidTranscriber:
    """Returns a result the database will refuse. Pydantic already rejects
    end < start, so model_construct() is used to skip validation and let the
    bad value reach the `ck_transcript_segments_time_range` CHECK constraint
    - i.e. to fail *inside* complete_job()'s transaction, after the
    Transcript row has been written."""

    def transcribe(self, media_path):
        bad_segment = TranscriptionSegmentResult.model_construct(
            start=9.0, end=1.0, text="impossible timing", confidence=0.9, words=[]
        )
        return TranscriptionResult.model_construct(text="x", segments=[bad_segment])


def _job(job_id: UUID) -> TranscriptionJob:
    with SessionLocal() as db:
        job = db.get(TranscriptionJob, job_id)
        db.expunge(job)
        return job


def test_a_transcriber_failure_marks_the_job_failed_with_a_generic_message(
    client, student, empty_job_queue
):
    job_id = _queue_job(client, student, "Failing transcriber")

    assert process_next_job(_RaisingTranscriber()) is True

    job = _job(job_id)
    assert job.status == "FAILED"
    assert job.transcript_id is None
    assert "boom" not in (job.error_message or ""), (
        "an internal exception message leaked into the user-visible error"
    )


def test_a_write_failing_mid_transaction_leaves_no_partial_transcript(
    client, student, empty_job_queue
):
    """complete_job() flushes the Transcript before inserting segments. When
    a segment violates a CHECK constraint the whole transaction must roll
    back - a Transcript row with no segments, attached to a job that says
    COMPLETED, would be exactly the corruption this guards against."""

    job_id = _queue_job(client, student, "Atomicity check")

    assert process_next_job(_DbInvalidTranscriber()) is True

    job = _job(job_id)
    assert job.status == "FAILED", "the job should be reported as failed, not completed"
    assert job.transcript_id is None

    with SessionLocal() as db:
        orphans = (
            db.query(Transcript)
            .filter(Transcript.owner_id == student.user_id, Transcript.title == "Atomicity check")
            .count()
        )
    assert orphans == 0, "a partially written Transcript survived the failed transaction"


def test_a_missing_media_file_fails_the_job_cleanly(client, student, empty_job_queue):
    """If the uploaded file has vanished from disk by the time the worker
    reaches it, that is a FileNotFoundError path with its own message."""

    from app.models.media import MediaFile
    from app.services.media_storage_service import resolve_stored_media

    job_id = _queue_job(client, student, "Vanished media")

    with SessionLocal() as db:
        job = db.get(TranscriptionJob, job_id)
        media = db.get(MediaFile, job.media_id)
        resolve_stored_media(media.storage_path).unlink()

    from app.transcribers.fake import FakeTranscriber

    assert process_next_job(FakeTranscriber()) is True

    job = _job(job_id)
    assert job.status == "FAILED"
    assert "could not be found" in (job.error_message or "")
