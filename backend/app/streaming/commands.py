"""Voice-command matching: fuzzy-match ASR text against a fixed phrase
vocabulary so students who cannot type can control the app by speaking.

Only used in COMMAND streaming mode. Dictation mode must never import or
apply anything from this module - normal speech must not be biased toward
the command vocabulary (see `StreamingTranscriber.transcribe`).
"""

import logging
from dataclasses import dataclass
from unicodedata import category

from rapidfuzz import fuzz

from app.core.config import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceCommand:
    id: str
    phrase: str
    destructive: bool = False


# Starter vocabulary - extend as more actions in the app become voice-
# controllable. Each phrase should be the shortest, most natural way a
# student would say the action out loud.
COMMANDS: tuple[VoiceCommand, ...] = (
    VoiceCommand(id="next", phrase="ඊළඟට"),
    VoiceCommand(id="previous", phrase="ආපසු"),
    VoiceCommand(id="stop", phrase="නවත්වන්න"),
    VoiceCommand(id="save", phrase="සුරකින්න"),
    VoiceCommand(id="submit", phrase="ඉදිරිපත් කරන්න", destructive=True),
    VoiceCommand(id="delete", phrase="මකන්න", destructive=True),
)

# Space-joined phrase list handed to faster-whisper's decoding bias
# (`hotwords`/`initial_prompt`) in command mode. See inference.py.
HOTWORDS = " ".join(command.phrase for command in COMMANDS)


def skeleton(text: str) -> str:
    """Strip Sinhala dependent vowel signs/virama and whitespace.

    Short command phrases fail most often because Whisper guesses the
    wrong vowel sign on an otherwise-correct consonant, e.g. "සුරකින්න"
    transcribed as "සුරකිනිනා". Comparing on this consonant-only skeleton
    lets those near-misses still match instead of requiring exact text
    equality.
    """

    lowered = text.strip().lower()
    without_diacritics = "".join(
        ch for ch in lowered if category(ch) not in ("Mn", "Mc")
    )
    return "".join(without_diacritics.split())


_SKELETON_BY_COMMAND_ID = {command.id: skeleton(command.phrase) for command in COMMANDS}


@dataclass(frozen=True)
class CommandMatch:
    command: VoiceCommand
    score: float


def match_command(
    transcript: str,
    *,
    avg_logprob: float | None = None,
    threshold: float | None = None,
    destructive_threshold: float | None = None,
    logprob_floor: float | None = None,
) -> CommandMatch | None:
    """Fuzzy-match a transcript against the known command vocabulary.

    Returns `None` rather than a low-confidence guess when nothing clears
    its threshold - callers should treat that as "no command was spoken",
    not fall back to the closest phrase regardless of how weak the match is.
    """

    threshold = settings.voice_command_fuzzy_threshold if threshold is None else threshold
    destructive_threshold = (
        settings.voice_command_destructive_threshold
        if destructive_threshold is None
        else destructive_threshold
    )
    logprob_floor = (
        settings.voice_command_logprob_floor if logprob_floor is None else logprob_floor
    )

    query = skeleton(transcript)
    best_command: VoiceCommand | None = None
    best_score = 0.0

    if query:
        for command in COMMANDS:
            score = fuzz.ratio(query, _SKELETON_BY_COMMAND_ID[command.id])
            if score > best_score:
                best_score = score
                best_command = command

    result: CommandMatch | None = None

    if best_command is not None:
        required = destructive_threshold if best_command.destructive else threshold
        # A borderline fuzzy score combined with a low-confidence ASR
        # result is too weak a signal on its own - the transcription
        # itself must have been reasonably confident too, unless the text
        # match was strong enough to stand on its own.
        weak_asr = avg_logprob is not None and avg_logprob < logprob_floor and best_score < 95

        if best_score >= required and not weak_asr:
            result = CommandMatch(command=best_command, score=best_score)

    logger.info(
        "voice_command_attempt transcript=%r matched=%s score=%.1f avg_logprob=%s",
        transcript,
        result.command.id if result else None,
        best_score,
        f"{avg_logprob:.3f}" if avg_logprob is not None else None,
    )

    return result
