from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile

from app.core.config import settings


AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".oga",
    ".webm",
    ".flac",
    ".aac",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".webm",
    ".mkv",
}

CHUNK_SIZE = 1024 * 1024


class MediaStorageError(ValueError):
    pass


class UnsupportedMediaError(MediaStorageError):
    pass


class UploadTooLargeError(MediaStorageError):
    pass


class EmptyUploadError(MediaStorageError):
    pass


@dataclass(frozen=True)
class StoredMedia:
    original_filename: str
    storage_path: str
    media_type: str
    file_size: int


def normalize_original_filename(filename: str | None) -> str:
    normalized = (filename or "upload").replace("\\", "/").split("/")[-1]

    if len(normalized) > 255:
        normalized = normalized[-255:]

    return normalized


def validate_upload(
    upload: UploadFile,
    transcript_type: str,
) -> tuple[str, str]:
    original_filename = normalize_original_filename(upload.filename)
    extension = Path(original_filename).suffix.lower()
    media_type = (upload.content_type or "").split(";")[0].strip().lower()

    is_audio = media_type.startswith("audio/")
    is_video = media_type.startswith("video/")

    if not is_audio and not is_video:
        raise UnsupportedMediaError(
            "Choose a supported audio or video file."
        )

    if is_audio and extension not in AUDIO_EXTENSIONS:
        raise UnsupportedMediaError(
            "The audio file extension is not supported."
        )

    if is_video and extension not in VIDEO_EXTENSIONS:
        raise UnsupportedMediaError(
            "The video file extension is not supported."
        )

    if transcript_type in {"NOTE", "QUIZ_ANSWER"} and not is_audio:
        raise UnsupportedMediaError(
            "Study notes and quiz answers require an audio file."
        )

    return extension, media_type


def save_media_upload(
    upload: UploadFile,
    media_id: UUID,
    transcript_type: str,
) -> StoredMedia:
    extension, media_type = validate_upload(
        upload=upload,
        transcript_type=transcript_type,
    )

    original_filename = normalize_original_filename(upload.filename)
    upload_directory = settings.media_storage_path
    upload_directory.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{media_id}{extension}"
    destination = upload_directory / stored_filename
    total_size = 0

    try:
        with destination.open("xb") as output_file:
            while chunk := upload.file.read(CHUNK_SIZE):
                total_size += len(chunk)

                if total_size > settings.max_upload_size_bytes:
                    raise UploadTooLargeError(
                        "The file is larger than the 100 MB limit."
                    )

                output_file.write(chunk)

        if total_size == 0:
            raise EmptyUploadError(
                "The uploaded file is empty."
            )

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        upload.file.close()

    return StoredMedia(
        original_filename=original_filename,
        storage_path=stored_filename,
        media_type=media_type,
        file_size=total_size,
    )


def delete_stored_media(storage_path: str) -> None:
    upload_directory = settings.media_storage_path
    destination = (upload_directory / storage_path).resolve()

    if destination.parent != upload_directory:
        raise MediaStorageError(
            "Invalid media storage path."
        )

    destination.unlink(missing_ok=True)


def resolve_stored_media(storage_path: str) -> Path:
    upload_directory = settings.media_storage_path.resolve()
    destination = (upload_directory / storage_path).resolve()

    if destination.parent != upload_directory:
        raise MediaStorageError(
            "Invalid media storage path."
        )

    if not destination.is_file():
        raise FileNotFoundError(
            "Stored media file was not found."
        )

    return destination
