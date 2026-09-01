from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.media import MediaFile
from app.models.transcription import TranscriptionJob
from app.models.user import User
from app.schemas.transcription import (
    TranscriptionJobCreatedResponse,
    TranscriptionJobStatusResponse,
)
from app.services.media_storage_service import (
    StoredMedia,
    delete_stored_media,
    save_media_upload,
)
from app.services.transcript_service import title_is_taken


ALLOWED_TRANSCRIPT_TYPES = {
    "LECTURE",
    "NOTE",
    "QUIZ_ANSWER",
}


class InvalidTranscriptionRequestError(ValueError):
    pass


def _pending_job_title_is_taken(db: Session, owner_id: UUID, title: str) -> bool:
    """A job still QUEUED/PROCESSING has no Transcript row yet, but will
    claim this title once it completes - so it counts as taken too."""

    statement = select(TranscriptionJob.job_id).where(
        TranscriptionJob.requested_by == owner_id,
        TranscriptionJob.status.in_(("QUEUED", "PROCESSING")),
        func.lower(TranscriptionJob.title) == title.strip().lower(),
    )

    return db.scalar(statement) is not None


def create_transcription_job(
    db: Session,
    user: User,
    upload: UploadFile,
    title: str,
    transcript_type: str,
) -> TranscriptionJob:
    normalized_title = title.strip()
    normalized_type = transcript_type.strip().upper()

    if not normalized_title:
        raise InvalidTranscriptionRequestError(
            "Transcript title is required."
        )

    if len(normalized_title) > 255:
        raise InvalidTranscriptionRequestError(
            "Transcript title cannot exceed 255 characters."
        )

    if normalized_type not in ALLOWED_TRANSCRIPT_TYPES:
        raise InvalidTranscriptionRequestError(
            "Invalid transcript type."
        )

    if title_is_taken(db, user.user_id, normalized_title) or _pending_job_title_is_taken(
        db, user.user_id, normalized_title
    ):
        raise InvalidTranscriptionRequestError(
            f'You already have a transcript titled "{normalized_title}".'
        )

    media_id = uuid4()
    stored_media: StoredMedia = save_media_upload(
        upload=upload,
        media_id=media_id,
        transcript_type=normalized_type,
    )

    try:
        media_file = MediaFile(
            media_id=media_id,
            owner_id=user.user_id,
            file_name=stored_media.original_filename,
            media_type=stored_media.media_type,
            storage_path=stored_media.storage_path,
            file_size=stored_media.file_size,
            duration_seconds=None,
        )

        job = TranscriptionJob(
            media_id=media_id,
            requested_by=user.user_id,
            transcript_id=None,
            title=normalized_title,
            transcript_type=normalized_type,
            status="QUEUED",
            progress_percent=0,
            error_message=None,
        )

        db.add(media_file)
        db.add(job)
        db.commit()
        db.refresh(job)

        return job

    except Exception:
        db.rollback()
        delete_stored_media(stored_media.storage_path)
        raise


def get_owned_transcription_job(
    db: Session,
    user: User,
    job_id: UUID,
) -> TranscriptionJob | None:
    statement = select(TranscriptionJob).where(
        TranscriptionJob.job_id == job_id,
        TranscriptionJob.requested_by == user.user_id,
    )

    return db.scalar(statement)


def serialize_created_job(
    job: TranscriptionJob,
) -> TranscriptionJobCreatedResponse:
    return TranscriptionJobCreatedResponse(
        job_id=job.job_id,
        status="UPLOADING",
    )


def serialize_job_status(
    job: TranscriptionJob,
) -> TranscriptionJobStatusResponse:
    public_status = (
        "PROCESSING"
        if job.status == "QUEUED"
        else job.status
    )

    return TranscriptionJobStatusResponse(
        job_id=job.job_id,
        status=public_status,
        progress=job.progress_percent,
        transcript_id=(
            job.transcript_id
            if job.status == "COMPLETED"
            else None
        ),
        message=(
            job.error_message
            if job.status == "FAILED"
            else None
        ),
    )
