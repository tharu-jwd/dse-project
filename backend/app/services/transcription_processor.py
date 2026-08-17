import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.media import MediaFile
from app.models.transcription import (
    Transcript,
    TranscriptSegment,
    TranscriptionJob,
)
from app.services.media_storage_service import resolve_stored_media
from app.transcribers.base import Transcriber, TranscriptionResult


logger = logging.getLogger(__name__)


class InvalidJobStateError(RuntimeError):
    pass


def claim_next_job() -> UUID | None:
    """Atomically claim the oldest queued job."""

    with SessionLocal.begin() as db:
        statement = (
            select(TranscriptionJob)
            .where(TranscriptionJob.status == "QUEUED")
            .order_by(TranscriptionJob.created_at.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        )

        job = db.scalar(statement)

        if job is None:
            return None

        job.status = "PROCESSING"
        job.progress_percent = 10
        job.started_at = datetime.now(timezone.utc)
        job.error_message = None

        return job.job_id


def get_job_media_path(job_id: UUID) -> Path:
    """Resolve the media path without holding a DB connection."""

    with SessionLocal() as db:
        job = db.get(TranscriptionJob, job_id)

        if job is None:
            raise InvalidJobStateError(
                "Transcription job no longer exists."
            )

        if job.status != "PROCESSING":
            raise InvalidJobStateError(
                "Transcription job is not in PROCESSING state."
            )

        media_file = db.get(MediaFile, job.media_id)

        if media_file is None:
            raise FileNotFoundError(
                "Media record was not found."
            )

        storage_path = media_file.storage_path

    return resolve_stored_media(storage_path)


def complete_job(
    job_id: UUID,
    result: TranscriptionResult,
) -> UUID:
    """Create transcript data and complete the job atomically."""

    with SessionLocal.begin() as db:
        statement = (
            select(TranscriptionJob)
            .where(TranscriptionJob.job_id == job_id)
            .with_for_update()
        )
        job = db.scalar(statement)

        if job is None:
            raise InvalidJobStateError(
                "Transcription job no longer exists."
            )

        if job.status != "PROCESSING":
            raise InvalidJobStateError(
                "Only a processing job can be completed."
            )

        if job.transcript_id is not None:
            raise InvalidJobStateError(
                "The job already has a transcript."
            )

        transcript = Transcript(
            owner_id=job.requested_by,
            media_id=job.media_id,
            title=job.title,
            transcript_type=job.transcript_type,
            status="DRAFT",
            segments=[
                TranscriptSegment(
                    segment_order=index,
                    generated_text=segment.text,
                    edited_text=None,
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=segment.confidence,
                    word_metadata=[
                        word.model_dump()
                        for word in segment.words
                    ],
                )
                for index, segment in enumerate(result.segments)
            ],
        )

        db.add(transcript)
        db.flush()

        media_file = db.get(MediaFile, job.media_id)

        if media_file is not None:
            media_file.duration_seconds = max(
                segment.end
                for segment in result.segments
            )

        job.transcript_id = transcript.transcript_id
        job.status = "COMPLETED"
        job.progress_percent = 100
        job.completed_at = datetime.now(timezone.utc)
        job.error_message = None

        return transcript.transcript_id


def fail_job(
    job_id: UUID,
    public_message: str,
) -> None:
    with SessionLocal.begin() as db:
        statement = (
            select(TranscriptionJob)
            .where(TranscriptionJob.job_id == job_id)
            .with_for_update()
        )
        job = db.scalar(statement)

        if job is None or job.status == "COMPLETED":
            return

        job.status = "FAILED"
        job.progress_percent = 100
        job.completed_at = datetime.now(timezone.utc)
        job.error_message = public_message[:1000]


def process_next_job(
    transcriber: Transcriber,
) -> bool:
    """Process one queued job. Return False when the queue is empty."""

    job_id = claim_next_job()

    if job_id is None:
        return False

    logger.info("Processing transcription job %s", job_id)

    try:
        media_path = get_job_media_path(job_id)
        result = transcriber.transcribe(media_path)
        transcript_id = complete_job(job_id, result)

        logger.info(
            "Completed transcription job %s as transcript %s",
            job_id,
            transcript_id,
        )

    except FileNotFoundError:
        logger.exception(
            "Media was unavailable for transcription job %s",
            job_id,
        )
        fail_job(
            job_id,
            "The uploaded media file could not be found.",
        )

    except Exception:
        logger.exception(
            "Transcription job %s failed",
            job_id,
        )
        fail_job(
            job_id,
            "The recording could not be transcribed.",
        )

    return True
