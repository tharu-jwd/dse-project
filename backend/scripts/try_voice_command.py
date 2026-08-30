"""Manual test tool: run one audio clip through command-mode transcription
and print what it matched, without needing the live streaming UI.

Usage (from the repo root, with the backend venv active):

    python -m scripts.try_voice_command path/to/clip.wav
    python -m scripts.try_voice_command path/to/clip.wav --mode dictation

Record a short clip of yourself saying a command (voice memo app, phone,
whatever) and point this at the file. It prints the raw transcript, ASR
confidence, and the matched command (or "no match") - the exact same
information written to the `voice_command_attempt` log line, so you can
build a stock of real phrases and see which ones the matcher struggles
with before wiring this into the live app.
"""

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from app.streaming.buffer import SAMPLE_RATE
from app.streaming.commands import match_command
from app.streaming.inference import get_streaming_transcriber


def _load_as_16k_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        # Simple linear-interpolation resample - good enough for manual
        # testing, not intended for production audio processing.
        duration = len(audio) / sample_rate
        target_length = int(duration * SAMPLE_RATE)
        original_positions = np.linspace(0, len(audio) - 1, num=len(audio))
        target_positions = np.linspace(0, len(audio) - 1, num=target_length)
        audio = np.interp(target_positions, original_positions, audio).astype("float32")

    return audio


async def _run(path: Path, mode: str) -> None:
    transcriber = get_streaming_transcriber()
    audio = _load_as_16k_mono(path)

    result = await transcriber.transcribe(audio, mode=mode)

    print(f"\nfile:        {path}")
    print(f"mode:        {mode}")
    print(f"transcript:  {result.text!r}")
    print(f"avg_logprob: {result.avg_logprob:.3f}")

    if mode != "command":
        print("(command matching skipped - not in command mode)")
        return

    match = match_command(result.text, avg_logprob=result.avg_logprob)
    if match is None:
        print("matched:     no command (nothing cleared the threshold)")
    else:
        print(
            f"matched:     {match.command.id!r} "
            f"(phrase={match.command.phrase!r}, score={match.score:.1f}, "
            f"destructive={match.command.destructive})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=Path, help="Path to a WAV/FLAC/etc. clip.")
    parser.add_argument(
        "--mode",
        choices=["command", "dictation"],
        default="command",
        help="Transcription mode to use (default: command).",
    )
    arguments = parser.parse_args()

    if not arguments.audio_file.is_file():
        raise SystemExit(f"No such file: {arguments.audio_file}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    asyncio.run(_run(arguments.audio_file, arguments.mode))


if __name__ == "__main__":
    main()
