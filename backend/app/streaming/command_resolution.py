"""Combine the fuzzy-text command match (app.streaming.commands) with
the speaker-embedding match (app.streaming.embeddings) into one decision
for a finalized command-mode utterance.

This is the one module allowed to depend on both: `commands.py` only
ever changed here to add a `language` parameter selecting which of its
two fixed phrase sets to fuzzy-match against - the vocabulary itself
still isn't touched by anything in this module. `embeddings.py` stays
ignorant of the command vocabulary entirely (per "implement this
independent from the commands"). Everything that needs both signals at
once lives here instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core.config import settings
from app.streaming.commands import COMMANDS_BY_LANGUAGE, match_command
from app.streaming.embeddings import best_match


logger = logging.getLogger(__name__)

Outcome = Literal["execute", "confirm", "none"]

# A command id's destructive-ness is a property of the action, not its
# wording - "delete" is destructive whether the spoken phrase is Sinhala
# or English - so this is computed once across every language's list
# rather than needing a per-language copy.
_DESTRUCTIVE_IDS = frozenset(
    command.id
    for commands in COMMANDS_BY_LANGUAGE.values()
    for command in commands
    if command.destructive
)


@dataclass(frozen=True)
class CommandDecision:
    outcome: Outcome
    # Only set when outcome == "execute" - the single command to act on.
    command_id: str | None
    fuzzy_command_id: str | None
    fuzzy_score: float | None
    embedding_command_id: str | None
    embedding_score: float | None
    agreed: bool  # both signals independently strong AND pointed at the same command


def _strong_embedding_match(
    embedding: np.ndarray | None,
    bank: dict[str, list[np.ndarray]],
) -> tuple[str | None, float | None]:
    if embedding is None or not bank:
        return None, None

    match = best_match(embedding, bank, threshold=settings.voice_embedding_similarity_threshold)

    if match is None:
        return None, None

    # Destructive commands don't get a third signal to check - they must
    # clear a *higher* bar within this signal to count as strong at all.
    # A score that only cleared the normal threshold is treated the same
    # as no match here, same idea as commands.py's own destructive_threshold.
    if match.label in _DESTRUCTIVE_IDS and match.score < settings.voice_embedding_destructive_threshold:
        return None, match.score

    return match.label, match.score


def resolve_command(
    transcript: str,
    *,
    avg_logprob: float | None,
    embedding: np.ndarray | None,
    bank: dict[str, list[np.ndarray]],
    language: str = "si",
) -> CommandDecision:
    """Decide what to do about one finalized command-mode utterance.

    Combination rule (every row is deliberate, not implicit):
      strong fuzzy + strong embedding, SAME command       -> execute (agreed)
      strong on either side, other weak/absent            -> execute (that one)
      strong fuzzy + strong embedding, DIFFERENT commands  -> confirm, never guess
      neither strong, but a borderline candidate on either side -> confirm
      no candidate at all, on either side                 -> none (ordinary dictation)

    Destructive commands (submit, delete) don't get an extra confirmation
    step beyond this - they must clear the higher destructive threshold
    *within* whichever signal calls them strong (see commands.py's
    `voice_command_destructive_threshold` and this module's
    `voice_embedding_destructive_threshold`). Clearing only the normal
    threshold makes that signal count as "not strong" for a destructive
    command, which naturally routes it to "confirm" or "none" via the
    same rules above - no separate code path needed.

    Fallback: if embedding matching is switched off, or this student has
    no enrollment bank, this returns exactly what the fuzzy path alone
    would have decided before this feature existed - enrollment is an
    enhancement, never a requirement.
    """

    strong_fuzzy = match_command(transcript, avg_logprob=avg_logprob, language=language)
    fuzzy_id = strong_fuzzy.command.id if strong_fuzzy else None
    fuzzy_score = strong_fuzzy.score if strong_fuzzy else None

    if not settings.voice_command_embedding_matching_enabled or not bank:
        decision = CommandDecision(
            outcome="execute" if fuzzy_id else "none",
            command_id=fuzzy_id,
            fuzzy_command_id=fuzzy_id,
            fuzzy_score=fuzzy_score,
            embedding_command_id=None,
            embedding_score=None,
            agreed=False,
        )
        _log(transcript, decision)
        return decision

    embedding_id, embedding_score = _strong_embedding_match(embedding, bank)

    if fuzzy_id and embedding_id:
        if fuzzy_id == embedding_id:
            decision = CommandDecision(
                "execute", fuzzy_id, fuzzy_id, fuzzy_score, embedding_id, embedding_score, True
            )
        else:
            decision = CommandDecision(
                "confirm", None, fuzzy_id, fuzzy_score, embedding_id, embedding_score, False
            )
    elif fuzzy_id:
        decision = CommandDecision(
            "execute", fuzzy_id, fuzzy_id, fuzzy_score, embedding_id, embedding_score, False
        )
    elif embedding_id:
        decision = CommandDecision(
            "execute", embedding_id, fuzzy_id, fuzzy_score, embedding_id, embedding_score, False
        )
    else:
        decision = _resolve_borderline(
            transcript, avg_logprob, embedding, bank, fuzzy_score, embedding_score, language
        )

    _log(transcript, decision)
    return decision


def _resolve_borderline(
    transcript: str,
    avg_logprob: float | None,
    embedding: np.ndarray | None,
    bank: dict[str, list[np.ndarray]],
    fuzzy_score: float | None,
    embedding_score: float | None,
    language: str,
) -> CommandDecision:
    """Neither signal was strong enough to execute on its own. Distinguish
    "maybe said a command, unsure" (confirm) from "just ordinary
    dictation" (none) by checking for a borderline candidate on either
    side - without this, the low bar of ordinary fuzzy string matching
    would make almost every dictated sentence loosely resemble *some*
    command and trigger a confirmation prompt, which would make normal
    note-taking unusable.
    """

    loose_fuzzy = match_command(
        transcript,
        avg_logprob=avg_logprob,
        threshold=settings.voice_command_fuzzy_borderline_floor,
        destructive_threshold=settings.voice_command_fuzzy_borderline_floor,
        language=language,
    )
    loose_embedding = (
        best_match(embedding, bank, threshold=settings.voice_embedding_borderline_floor)
        if embedding is not None and bank
        else None
    )

    if loose_fuzzy is None and loose_embedding is None:
        return CommandDecision("none", None, None, fuzzy_score, None, embedding_score, False)

    return CommandDecision(
        "confirm",
        None,
        loose_fuzzy.command.id if loose_fuzzy else None,
        loose_fuzzy.score if loose_fuzzy else fuzzy_score,
        loose_embedding.label if loose_embedding else None,
        loose_embedding.score if loose_embedding else embedding_score,
        False,
    )


def _log(transcript: str, decision: CommandDecision) -> None:
    logger.info(
        "voice_command_decision transcript=%r outcome=%s command=%s "
        "fuzzy=%s(%s) embedding=%s(%s) agreed=%s",
        transcript,
        decision.outcome,
        decision.command_id,
        decision.fuzzy_command_id,
        f"{decision.fuzzy_score:.1f}" if decision.fuzzy_score is not None else None,
        decision.embedding_command_id,
        f"{decision.embedding_score:.3f}" if decision.embedding_score is not None else None,
        decision.agreed,
    )
