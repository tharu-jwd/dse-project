"""One-off: push the raw clips in storage/voice_samples/ into a student's
real CommandEnrollment bank, replacing whatever was there before.

The normal enrollment path is one-clip-at-a-time from the browser
(app/api/routes/voice_enrollment.py). This script exists for the case
where samples were already collected in bulk via the /dev/voice-samples
tool and need to become the live enrollment without re-recording through
the UI. Takes at most `voice_enrollment_samples_required` clips per
command (sorted by take number) to match what the app itself would ever
store.

Usage (from backend/, with the venv active):
    python -m scripts.reimport_voice_enrollment student@sinhaspeech.lk
"""

import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.user import User
from app.models.voice_enrollment import CommandEnrollment
from app.streaming.commands import COMMANDS
from app.streaming.embeddings import embed_audio
from app.streaming.inference import get_streaming_transcriber

SAMPLE_RATE = 16_000
FILENAME_RE = re.compile(r"^(?P<command_id>.+)_(?P<n>\d+)\.wav$", re.IGNORECASE)
_COMMAND_IDS = {command.id for command in COMMANDS}


def load_16k_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        duration = len(audio) / sample_rate
        target_length = int(duration * SAMPLE_RATE)
        original_positions = np.linspace(0, len(audio) - 1, num=len(audio))
        target_positions = np.linspace(0, len(audio) - 1, num=target_length)
        audio = np.interp(target_positions, original_positions, audio).astype("float32")
    return audio


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m scripts.reimport_voice_enrollment <user-email>")

    email = sys.argv[1]
    wav_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("storage/voice_samples")
    required = settings.voice_enrollment_samples_required

    by_command: dict[str, list[Path]] = {}
    for wav_path in sorted(wav_dir.glob("*.wav")):
        match = FILENAME_RE.match(wav_path.name)
        if not match or match.group("command_id") not in _COMMAND_IDS:
            continue
        by_command.setdefault(match.group("command_id"), []).append(
            (int(match.group("n")), wav_path)
        )

    model = get_streaming_transcriber()._model

    with SessionLocal.begin() as db:
        user = db.query(User).filter(User.email == email).one_or_none()
        if user is None:
            raise SystemExit(f"no user with email {email!r}")

        for command_id, takes in by_command.items():
            takes.sort(key=lambda pair: pair[0])
            takes = takes[:required]

            db.query(CommandEnrollment).filter(
                CommandEnrollment.user_id == user.user_id,
                CommandEnrollment.command_id == command_id,
            ).delete()

            for sample_index, (_, wav_path) in enumerate(takes):
                audio = load_16k_mono(wav_path)
                embedding = embed_audio(model, audio)
                db.add(
                    CommandEnrollment(
                        user_id=user.user_id,
                        command_id=command_id,
                        sample_index=sample_index,
                        embedding=embedding.astype(np.float32).tobytes(),
                        embedding_dim=int(embedding.shape[0]),
                        model_version=settings.voice_embedding_model_version,
                    )
                )
            print(f"{command_id}: replaced with {len(takes)} sample(s) from {wav_dir}")

    print(f"done - enrollment for {email} now reflects {wav_dir}")


if __name__ == "__main__":
    main()
