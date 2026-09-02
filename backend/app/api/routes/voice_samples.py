"""Step-1 data collection for the voice-command embedding work.

Lets the app itself record command-phrase samples straight from the
browser and save them as `{command_id}_{n}.wav`, so
`scripts/validate_command_embeddings.py` has something real to measure
instead of requiring a manual file transfer.

This is NOT the per-student enrollment bank described in the wider task -
that's a later step, gated on what this data tells us. There is
deliberately no CommandEnrollment table or matching logic here yet.
"""

import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, UploadFile, status

from app.api.dependencies import CurrentUserDependency
from app.core.config import settings
from app.streaming.commands import COMMANDS


router = APIRouter(prefix="/voice-samples", tags=["Voice samples (dev)"])

_COMMAND_IDS = {command.id for command in COMMANDS}

# `lang` picks which on-disk sample set a recording lands in, so trying an
# English phrase set never overwrites or mixes with the existing Sinhala
# recordings commands.py's current phrases were validated against.
_LANG_SUFFIX = {"si": "", "en": "_en"}


def _samples_dir(lang: str = "si") -> Path:
    suffix = _LANG_SUFFIX[lang]
    path = settings.voice_samples_path.with_name(settings.voice_samples_path.name + suffix)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _next_take(command_id: str, lang: str) -> int:
    numbers = []

    for existing in _samples_dir(lang).glob(f"{command_id}_*.wav"):
        try:
            numbers.append(int(existing.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue

    return max(numbers) + 1 if numbers else 1


def _counts(lang: str) -> dict[str, int]:
    counts = {command_id: 0 for command_id in _COMMAND_IDS}

    for existing in _samples_dir(lang).glob("*.wav"):
        command_id = existing.stem.rsplit("_", 1)[0]
        if command_id in counts:
            counts[command_id] += 1

    return counts


@router.get("")
def list_progress(
    _: CurrentUserDependency,
    lang: str = Query("si", pattern="^(si|en)$"),
) -> dict:
    counts = _counts(lang)
    return {
        "commands": [
            {"id": command.id, "phrase": command.phrase, "count": counts[command.id]}
            for command in COMMANDS
        ]
    }


@router.post("/{command_id}")
async def upload_sample(
    command_id: str,
    file: UploadFile,
    _: CurrentUserDependency,
    lang: str = Query("si", pattern="^(si|en)$"),
) -> dict:
    if command_id not in _COMMAND_IDS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown command id.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="The recording was empty.")

    take = _next_take(command_id, lang)
    destination = _samples_dir(lang) / f"{command_id}_{take}.wav"
    suffix = Path(file.filename or "clip.webm").suffix or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix) as raw_upload:
        raw_upload.write(raw)
        raw_upload.flush()

        # Whatever format the browser's MediaRecorder produced (webm/opus,
        # typically), converted to the plain 16kHz mono WAV the validation
        # harness (and the real streaming path) expect.
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_upload.name,
                "-ar",
                "16000",
                "-ac",
                "1",
                str(destination),
            ],
            capture_output=True,
        )

    if result.returncode != 0 or not destination.exists():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The recording could not be converted. Please try again.",
        )

    return {"command_id": command_id, "take": take, "counts": _counts(lang)}


@router.delete("/{command_id}")
def delete_samples(
    command_id: str,
    _: CurrentUserDependency,
    lang: str = Query("si", pattern="^(si|en)$"),
) -> dict:
    if command_id not in _COMMAND_IDS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown command id.")

    for existing in _samples_dir(lang).glob(f"{command_id}_*.wav"):
        existing.unlink()

    return {"command_id": command_id, "counts": _counts(lang)}
