"""Push raw clips from storage/voice_samples/ into a student's real
CommandEnrollment bank for one language, replacing what was there.

The normal enrollment path is one clip at a time from the browser
(app/api/routes/voice_enrollment.py). This script is for clips already
collected in bulk that should become the live enrollment without
re-recording. Takes at most `voice_enrollment_samples_required` clips per
command, sorted by take number.

Usage (from backend/, with the venv active):
    python -m scripts.reimport_voice_enrollment student@sinhaspeech.lk \
        --commands next,previous,delete,submit,save --language si --reset-others

--commands      only import these command ids (default: every id with clips)
--reset-others  also delete this language's samples for every other
                enrollable command (including the wake word), so they can
                be re-recorded from scratch in the app
--dir           clip directory (default: ../storage/voice_samples)
"""

import argparse
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
from app.streaming.commands import enrollment_commands
from app.streaming.embeddings import embed_audio
from app.streaming.inference import get_streaming_transcriber

SAMPLE_RATE = 16_000
FILENAME_RE = re.compile(r"^(?P<command_id>.+)_(?P<n>\d+)\.wav$", re.IGNORECASE)
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "storage" / "voice_samples"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("email")
    parser.add_argument("--language", default="si", choices=("si", "en"))
    parser.add_argument("--commands", default="")
    parser.add_argument("--reset-others", action="store_true")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    valid_ids = {command.id for command in enrollment_commands(args.language)}
    wanted = {c.strip() for c in args.commands.split(",") if c.strip()} or valid_ids
    unknown = wanted - valid_ids
    if unknown:
        raise SystemExit(f"unknown command id(s) for {args.language!r}: {sorted(unknown)}")

    required = settings.voice_enrollment_samples_required
    by_command: dict[str, list[tuple[int, Path]]] = {}
    for wav_path in sorted(args.dir.glob("*.wav")):
        match = FILENAME_RE.match(wav_path.name)
        if match and match.group("command_id") in wanted:
            by_command.setdefault(match.group("command_id"), []).append(
                (int(match.group("n")), wav_path)
            )

    missing = wanted - by_command.keys()
    if missing:
        raise SystemExit(f"no clips found in {args.dir} for: {sorted(missing)}")

    model = get_streaming_transcriber()._model

    with SessionLocal.begin() as db:
        user = db.query(User).filter(User.email == args.email).one_or_none()
        if user is None:
            raise SystemExit(f"no user with email {args.email!r}")

        in_language = db.query(CommandEnrollment).filter(
            CommandEnrollment.user_id == user.user_id,
            CommandEnrollment.language == args.language,
        )

        if args.reset_others:
            removed = in_language.filter(CommandEnrollment.command_id.notin_(wanted)).delete(
                synchronize_session=False
            )
            print(f"reset: deleted {removed} {args.language} sample(s) outside {sorted(wanted)}")

        for command_id, takes in sorted(by_command.items()):
            takes = sorted(takes)[:required]
            in_language.filter(CommandEnrollment.command_id == command_id).delete(
                synchronize_session=False
            )

            for sample_index, (_, wav_path) in enumerate(takes):
                embedding = embed_audio(model, load_16k_mono(wav_path))
                db.add(
                    CommandEnrollment(
                        user_id=user.user_id,
                        command_id=command_id,
                        language=args.language,
                        sample_index=sample_index,
                        embedding=embedding.astype(np.float32).tobytes(),
                        embedding_dim=int(embedding.shape[0]),
                        model_version=settings.voice_embedding_model_version,
                    )
                )
            print(f"{command_id}: {len(takes)} sample(s) from {[p.name for _, p in takes]}")

    print(f"done - {args.email} ({args.language})")


if __name__ == "__main__":
    main()
