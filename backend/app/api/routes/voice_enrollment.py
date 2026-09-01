"""Enrollment endpoints for speaker-embedded voice commands (step 3).

Resumable by design: a student can submit however many samples they
have time for in one sitting and come back later - nothing here
requires all six commands to be completed in one request or one
session. See app.services.voice_enrollment for the storage/validation
logic and app.streaming.embeddings for why command_id is treated as an
opaque string throughout.
"""

import asyncio
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from app.api.dependencies import CurrentUserDependency, SessionDependency
from app.services import voice_enrollment
from app.streaming.embeddings import ClipTooShortError
from app.streaming.inference import get_streaming_transcriber


router = APIRouter(prefix="/voice-enrollment", tags=["Voice enrollment"])

SAMPLE_RATE = 16_000
LanguageQuery = Query("si", pattern="^(si|en)$")


def _decode_upload_to_16k_mono(raw: bytes, filename: str) -> np.ndarray:
    """Whatever format the browser's MediaRecorder produced (webm/opus,
    typically) -> 16kHz mono float32, the format the streaming path and
    the embedding function both expect."""

    suffix = Path(filename or "clip.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix) as raw_upload:
        raw_upload.write(raw)
        raw_upload.flush()

        with tempfile.NamedTemporaryFile(suffix=".wav") as converted:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    raw_upload.name,
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    converted.name,
                ],
                capture_output=True,
            )

            if result.returncode != 0:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="The recording could not be read. Please try again.",
                )

            audio, sample_rate = sf.read(converted.name, dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    assert sample_rate == SAMPLE_RATE  # ffmpeg was told to resample to this

    return audio


@router.get("")
def get_status(user: CurrentUserDependency, language: str = LanguageQuery) -> dict:
    progress = voice_enrollment.get_progress(user.user_id, language)
    return {
        "language": language,
        "activeLanguage": user.command_language,
        "commands": [
            {
                "id": item.command_id,
                "phrase": item.phrase,
                "destructive": item.destructive,
                "required": item.required,
                "collected": item.collected,
                "complete": item.complete,
            }
            for item in progress
        ],
    }


@router.patch("/active-language")
def set_active_language(
    payload: dict,
    user: CurrentUserDependency,
    db: SessionDependency,
) -> dict:
    """Which phrase set is "live" for this student at runtime - a data
    setting on the user row, never a code change. Enrollment samples for
    both languages stay untouched either way; this only decides which
    one `load_bank`/`resolve_command` reach for during a live session."""

    language = payload.get("language")
    if language not in ("si", "en"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="language must be 'si' or 'en'."
        )

    user.command_language = language
    db.commit()

    return {"activeLanguage": user.command_language}


@router.post("/{command_id}/samples")
async def submit_sample(
    command_id: str,
    file: UploadFile,
    user: CurrentUserDependency,
    language: str = LanguageQuery,
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="The recording was empty.")

    audio = await asyncio.to_thread(_decode_upload_to_16k_mono, raw, file.filename or "")

    try:
        embedding = await get_streaming_transcriber().embed(audio)
    except ClipTooShortError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    try:
        result = await asyncio.to_thread(
            voice_enrollment.submit_sample, user.user_id, command_id, embedding, language
        )
    except voice_enrollment.UnknownCommandError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    return {
        "accepted": result.accepted,
        "reason": result.reason,
        "similarity": result.similarity,
        "collected": result.collected,
        "required": result.required,
    }


@router.post("/{command_id}/practice")
async def practice_sample(
    command_id: str,
    file: UploadFile,
    user: CurrentUserDependency,
    language: str = LanguageQuery,
) -> dict:
    """Practice mode: check a fresh attempt against this student's own
    enrolled samples without executing anything - purely a "how did I
    do" preview, so a student can rehearse before relying on a command
    live."""

    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="The recording was empty.")

    audio = await asyncio.to_thread(_decode_upload_to_16k_mono, raw, file.filename or "")

    try:
        embedding = await get_streaming_transcriber().embed(audio)
    except ClipTooShortError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    try:
        result = await asyncio.to_thread(
            voice_enrollment.practice_command, user.user_id, command_id, embedding, language
        )
    except voice_enrollment.UnknownCommandError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    return {
        "commandId": result.command_id,
        "ownSimilarity": result.own_similarity,
        "passesThreshold": result.passes_threshold,
        "enrolledSampleCount": result.enrolled_sample_count,
        "closestOtherCommandId": result.closest_other_command_id,
        "closestOtherSimilarity": result.closest_other_similarity,
    }


@router.delete("/{command_id}")
def delete_samples(
    command_id: str, user: CurrentUserDependency, language: str = LanguageQuery
) -> dict:
    try:
        voice_enrollment.delete_samples(user.user_id, command_id, language)
    except voice_enrollment.UnknownCommandError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(error)) from error

    progress = next(
        item
        for item in voice_enrollment.get_progress(user.user_id, language)
        if item.command_id == command_id
    )
    return {"command_id": command_id, "collected": progress.collected, "required": progress.required}
