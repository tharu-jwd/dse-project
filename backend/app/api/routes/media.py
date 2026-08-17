from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.api.dependencies import (
    CurrentUserDependency,
    SessionDependency,
)
from app.services.media_access_service import get_accessible_media
from app.services.media_storage_service import (
    MediaStorageError,
    resolve_stored_media,
)


router = APIRouter(
    prefix="/media",
    tags=["Media"],
)


@router.get(
    "/{media_id}",
    response_class=FileResponse,
    name="get_media",
)
def get_media(
    media_id: UUID,
    db: SessionDependency,
    current_user: CurrentUserDependency,
) -> FileResponse:
    media_file = get_accessible_media(
        db=db,
        user=current_user,
        media_id=media_id,
    )

    if media_file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media file was not found.",
        )

    try:
        media_path = resolve_stored_media(
            media_file.storage_path,
        )
    except (FileNotFoundError, MediaStorageError) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media file was not found.",
        ) from error

    return FileResponse(
        path=media_path,
        media_type=media_file.media_type,
        filename=media_file.file_name,
        content_disposition_type="inline",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Accept-Ranges": "bytes",
        },
    )
