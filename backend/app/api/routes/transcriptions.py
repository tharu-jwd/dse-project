from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.api.dependencies import (
    CurrentUserDependency,
    SessionDependency,
)
from app.schemas.transcription import (
    TranscriptionJobCreatedResponse,
    TranscriptionJobStatusResponse,
)
from app.services.media_storage_service import (
    EmptyUploadError,
    UnsupportedMediaError,
    UploadTooLargeError,
)
from app.services.transcription_job_service import (
    InvalidTranscriptionRequestError,
    create_transcription_job,
    get_owned_transcription_job,
    serialize_created_job,
    serialize_job_status,
)


router = APIRouter(
    prefix="/transcriptions",
    tags=["Transcriptions"],
)


@router.post(
    "",
    response_model=TranscriptionJobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_transcription(
    db: SessionDependency,
    current_user: CurrentUserDependency,
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form()],
    transcript_type: Annotated[str, Form(alias="type")],
) -> TranscriptionJobCreatedResponse:
    try:
        job = create_transcription_job(
            db=db,
            user=current_user,
            upload=file,
            title=title,
            transcript_type=transcript_type,
        )

    except InvalidTranscriptionRequestError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    except UnsupportedMediaError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error

    except UploadTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(error),
        ) from error

    except EmptyUploadError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return serialize_created_job(job)


@router.get(
    "/jobs/{job_id}",
    response_model=TranscriptionJobStatusResponse,
)
def get_transcription_job(
    job_id: UUID,
    db: SessionDependency,
    current_user: CurrentUserDependency,
) -> TranscriptionJobStatusResponse:
    job = get_owned_transcription_job(
        db=db,
        user=current_user,
        job_id=job_id,
    )

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcription job was not found.",
        )

    return serialize_job_status(job)
