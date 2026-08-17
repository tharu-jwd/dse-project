from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.models.media import MediaFile
from app.models.transcription import Transcript
from app.models.user import User


def get_accessible_media(
    db: Session,
    user: User,
    media_id: UUID,
) -> MediaFile | None:
    media_file = db.get(MediaFile, media_id)

    if media_file is None:
        return None

    if media_file.owner_id == user.user_id:
        return media_file

    if user.role != "TEACHER":
        return None

    teacher_can_access = db.scalar(
        select(
            exists().where(
                Transcript.media_id == media_id,
                Transcript.transcript_type == "LECTURE",
            )
        )
    )

    if not teacher_can_access:
        return None

    return media_file
