"""Enrollment storage for speaker-embedded voice commands.

This is the one place allowed to know both about `app.streaming.commands`
(to know which commands exist and need prompting for) and about
`app.streaming.embeddings` (the vector maths). `embeddings.py` itself
stays ignorant of the command vocabulary; this module is what connects
"command id" (a plain string chosen by `commands.py`, today) to "a bank
of embeddings for it" - editing that vocabulary never requires a change
here, it just changes what `get_progress`/`submit_sample` iterate over
next time they run.

Every function here takes a `language` ("si" or "en") because
`CommandEnrollment` rows are scoped per-language, not just per command
id - the same id ("delete") names a different phrase in each language,
so its samples live in entirely separate rows. Deleting and re-recording
a command's samples (`delete_samples` then `submit_sample`) is a pure
data operation - it never touches this module or commands.py.
"""

import logging
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.voice_enrollment import CommandEnrollment
from app.streaming.commands import COMMANDS_BY_LANGUAGE, get_commands
from app.streaming.embeddings import manhattan_similarity


logger = logging.getLogger(__name__)


def _valid_command_ids(language: str) -> set[str]:
    # Deliberately strict here, unlike get_commands()'s Sinhala fallback -
    # that fallback exists so a bad language value degrades runtime
    # matching gracefully; a write path like enrollment should reject an
    # invalid language outright rather than silently storing samples
    # against the wrong list, or worse, tripping the DB's own
    # ck_command_enrollments_language / ck_users_command_language
    # constraints with a raw IntegrityError instead of this clear error.
    return {command.id for command in COMMANDS_BY_LANGUAGE.get(language, ())}


class UnknownCommandError(ValueError):
    """Raised for a command_id not present in the current vocabulary."""


@dataclass(frozen=True)
class CommandProgress:
    command_id: str
    phrase: str
    destructive: bool
    required: int
    collected: int

    @property
    def complete(self) -> bool:
        return self.collected >= self.required


@dataclass(frozen=True)
class SubmitResult:
    accepted: bool
    reason: str | None  # None on success; else "already_complete" or "low_similarity"
    similarity: float | None  # best match against this command's existing samples
    collected: int
    required: int


@dataclass(frozen=True)
class PracticeResult:
    """One "does this sound like me?" check against a student's own
    enrollment bank - the same comparison the runtime matcher does, just
    surfaced as a practice/preview tool instead of an executed command."""

    command_id: str
    own_similarity: float | None  # best match against this command's own enrolled samples
    passes_threshold: bool  # would this clear the actual runtime match threshold
    enrolled_sample_count: int
    closest_other_command_id: str | None  # best-matching *different* enrolled command, if any
    closest_other_similarity: float | None


def practice_command(
    user_id: UUID, command_id: str, embedding: np.ndarray, language: str = "si"
) -> PracticeResult:
    """Compare a fresh attempt against a student's own enrollment bank,
    without executing anything - lets them check "how did I do?" and
    catch a command that's landing closer to a *different* enrolled
    command than to its own, before that becomes a real mismatch."""

    if command_id not in _valid_command_ids(language):
        raise UnknownCommandError(f"Unknown command id {command_id!r} for language {language!r}.")

    bank = load_bank(user_id, language)
    own_samples = bank.get(command_id, [])
    # manhattan_similarity's own arithmetic promotes a plain float through
    # a numpy division, so despite its `-> float` annotation it actually
    # returns numpy.float64 - float(...) here guarantees a native Python
    # float survives into the API response (numpy.bool_ from comparing
    # one directly against a threshold isn't JSON-serializable at all).
    own_similarity = (
        float(max(manhattan_similarity(embedding, sample) for sample in own_samples))
        if own_samples
        else None
    )

    closest_other_id: str | None = None
    closest_other_similarity: float | None = None
    for other_id, samples in bank.items():
        if other_id == command_id or not samples:
            continue
        best = float(max(manhattan_similarity(embedding, sample) for sample in samples))
        if closest_other_similarity is None or best > closest_other_similarity:
            closest_other_id = other_id
            closest_other_similarity = best

    passes_threshold = bool(
        own_similarity is not None
        and own_similarity >= settings.voice_embedding_similarity_threshold
    )

    return PracticeResult(
        command_id=command_id,
        own_similarity=own_similarity,
        passes_threshold=passes_threshold,
        enrolled_sample_count=len(own_samples),
        closest_other_command_id=closest_other_id,
        closest_other_similarity=closest_other_similarity,
    )


def get_progress(user_id: UUID, language: str = "si") -> list[CommandProgress]:
    """Per-command completeness for one language - backs both the
    "start/resume" and "status" endpoints, since resuming is just "show
    me status and let me carry on"."""

    required = settings.voice_enrollment_samples_required

    with SessionLocal() as db:
        rows = (
            db.query(CommandEnrollment.command_id)
            .filter(
                CommandEnrollment.user_id == user_id,
                CommandEnrollment.language == language,
            )
            .all()
        )

    counts: dict[str, int] = {}
    for (command_id,) in rows:
        counts[command_id] = counts.get(command_id, 0) + 1

    return [
        CommandProgress(
            command_id=command.id,
            phrase=command.phrase,
            destructive=command.destructive,
            required=required,
            collected=counts.get(command.id, 0),
        )
        for command in get_commands(language)
    ]


def submit_sample(
    user_id: UUID, command_id: str, embedding: np.ndarray, language: str = "si"
) -> SubmitResult:
    """Validate one enrollment recording's embedding and store it on success.

    A mis-recorded sample poisons the bank, so anything that doesn't
    resemble this command's own previously-accepted samples is rejected
    rather than stored - the caller (the route) asks the student to
    repeat it. The very first sample for a command has nothing to
    compare against yet and is always accepted.
    """

    if command_id not in _valid_command_ids(language):
        raise UnknownCommandError(f"Unknown command id {command_id!r} for language {language!r}.")

    required = settings.voice_enrollment_samples_required

    with SessionLocal.begin() as db:
        existing = (
            db.query(CommandEnrollment)
            .filter(
                CommandEnrollment.user_id == user_id,
                CommandEnrollment.command_id == command_id,
                CommandEnrollment.language == language,
            )
            .order_by(CommandEnrollment.sample_index)
            .all()
        )
        collected = len(existing)

        if collected >= required:
            return SubmitResult(
                accepted=False,
                reason="already_complete",
                similarity=None,
                collected=collected,
                required=required,
            )

        similarity: float | None = None
        if existing:
            similarity = max(
                manhattan_similarity(embedding, np.frombuffer(row.embedding, dtype=np.float32))
                for row in existing
            )
            if similarity < settings.voice_enrollment_min_sample_similarity:
                return SubmitResult(
                    accepted=False,
                    reason="low_similarity",
                    similarity=similarity,
                    collected=collected,
                    required=required,
                )

        next_index = max((row.sample_index for row in existing), default=-1) + 1
        db.add(
            CommandEnrollment(
                user_id=user_id,
                command_id=command_id,
                language=language,
                sample_index=next_index,
                embedding=embedding.astype(np.float32).tobytes(),
                embedding_dim=int(embedding.shape[0]),
                model_version=settings.voice_embedding_model_version,
            )
        )

    return SubmitResult(
        accepted=True,
        reason=None,
        similarity=similarity,
        collected=collected + 1,
        required=required,
    )


def delete_samples(user_id: UUID, command_id: str, language: str = "si") -> None:
    if command_id not in _valid_command_ids(language):
        raise UnknownCommandError(f"Unknown command id {command_id!r} for language {language!r}.")

    with SessionLocal.begin() as db:
        db.query(CommandEnrollment).filter(
            CommandEnrollment.user_id == user_id,
            CommandEnrollment.command_id == command_id,
            CommandEnrollment.language == language,
        ).delete()


def load_bank(user_id: UUID, language: str = "si") -> dict[str, list[np.ndarray]]:
    """A student's enrollment bank for one language, for runtime matching.

    Only the currently active language's samples are ever loaded - the
    other language's enrollments (if any) stay in the table untouched,
    so switching a student's active language back and forth never loses
    data or requires re-recording anything already done.

    Rows stamped with a `model_version` other than the currently
    configured checkpoint are skipped rather than included - an
    embedding produced by a different model isn't comparable to one
    from the current model, so silently mixing them in would make
    matching worse, not approximately right. See
    `settings.voice_embedding_model_version`.
    """

    current_version = settings.voice_embedding_model_version

    with SessionLocal() as db:
        rows = (
            db.query(CommandEnrollment)
            .filter(
                CommandEnrollment.user_id == user_id,
                CommandEnrollment.language == language,
            )
            .all()
        )

    bank: dict[str, list[np.ndarray]] = {}
    stale_count = 0

    for row in rows:
        if row.model_version != current_version:
            stale_count += 1
            continue

        bank.setdefault(row.command_id, []).append(
            np.frombuffer(row.embedding, dtype=np.float32)
        )

    if stale_count:
        logger.warning(
            "Skipped %d stale enrollment sample(s) for user %s "
            "(model_version mismatch, current=%r).",
            stale_count,
            user_id,
            current_version,
        )

    return bank
