"""Enrollment storage for speaker-embedded voice commands.

This is the one place allowed to know both about `app.streaming.commands`
(to know which commands exist and need prompting for) and about
`app.streaming.embeddings` (the vector maths). `embeddings.py` itself
stays ignorant of the command vocabulary; this module is what connects
"command id" (a plain string chosen by `commands.py`, today) to "a bank
of embeddings for it" - editing that vocabulary never requires a change
here, it just changes what `get_progress`/`submit_sample` iterate over
next time they run.
"""

import logging
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.voice_enrollment import CommandEnrollment
from app.streaming.commands import COMMANDS
from app.streaming.embeddings import cosine_similarity


logger = logging.getLogger(__name__)

_VALID_COMMAND_IDS = {command.id for command in COMMANDS}


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


def get_progress(user_id: UUID) -> list[CommandProgress]:
    """Per-command completeness - backs both the "start/resume" and
    "status" endpoints, since resuming is just "show me status and let
    me carry on"."""

    required = settings.voice_enrollment_samples_required

    with SessionLocal() as db:
        rows = (
            db.query(CommandEnrollment.command_id)
            .filter(CommandEnrollment.user_id == user_id)
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
        for command in COMMANDS
    ]


def submit_sample(user_id: UUID, command_id: str, embedding: np.ndarray) -> SubmitResult:
    """Validate one enrollment recording's embedding and store it on success.

    A mis-recorded sample poisons the bank, so anything that doesn't
    resemble this command's own previously-accepted samples is rejected
    rather than stored - the caller (the route) asks the student to
    repeat it. The very first sample for a command has nothing to
    compare against yet and is always accepted.
    """

    if command_id not in _VALID_COMMAND_IDS:
        raise UnknownCommandError(f"Unknown command id: {command_id!r}")

    required = settings.voice_enrollment_samples_required

    with SessionLocal.begin() as db:
        existing = (
            db.query(CommandEnrollment)
            .filter(
                CommandEnrollment.user_id == user_id,
                CommandEnrollment.command_id == command_id,
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
                cosine_similarity(embedding, np.frombuffer(row.embedding, dtype=np.float32))
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


def delete_samples(user_id: UUID, command_id: str) -> None:
    if command_id not in _VALID_COMMAND_IDS:
        raise UnknownCommandError(f"Unknown command id: {command_id!r}")

    with SessionLocal.begin() as db:
        db.query(CommandEnrollment).filter(
            CommandEnrollment.user_id == user_id,
            CommandEnrollment.command_id == command_id,
        ).delete()


def load_bank(user_id: UUID) -> dict[str, list[np.ndarray]]:
    """A student's full enrollment bank, for runtime matching.

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
            .filter(CommandEnrollment.user_id == user_id)
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
