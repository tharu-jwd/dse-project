"""Tests for the fuzzy+embedding combination logic - every row of the
combination table, the destructive-command bar, and the unenrolled/
disabled fallback. `match_command` and `best_match` are stubbed out
here since their own matching behaviour is already covered by
test_commands.py and test_embeddings.py; this file is purely about
`resolve_command`'s decision logic given controlled inputs from each.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from app.streaming import command_resolution as cr
from app.streaming.commands import VoiceCommand


def _command(command_id: str, destructive: bool = False) -> VoiceCommand:
    return VoiceCommand(id=command_id, phrase=command_id, destructive=destructive)


def _fuzzy(command_id: str, score: float, destructive: bool = False):
    return SimpleNamespace(command=_command(command_id, destructive), score=score)


def _embedding(label: str, score: float):
    return SimpleNamespace(label=label, score=score)


def _stub_match_command(strong=None, borderline=None):
    """threshold=None is the "strong" call (uses settings' real default);
    an explicit threshold is always the borderline call in this module."""

    def fake(transcript, *, avg_logprob=None, threshold=None, destructive_threshold=None, logprob_floor=None):
        return strong if threshold is None else borderline

    return fake


def _stub_best_match(strong=None, borderline=None):
    def fake(embedding, bank, *, threshold=None):
        if threshold == cr.settings.voice_embedding_similarity_threshold:
            return strong
        return borderline

    return fake


@pytest.fixture(autouse=True)
def _enable_embedding_matching(monkeypatch):
    monkeypatch.setattr(cr.settings, "voice_command_embedding_matching_enabled", True)


NON_EMPTY_BANK = {"delete": [np.zeros(4, dtype=np.float32)]}


def test_both_strong_same_command_executes_and_agrees(monkeypatch):
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=_fuzzy("delete", 95.0)))
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=_embedding("delete", 0.95)))

    decision = cr.resolve_command(
        "මකන්න", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK
    )

    assert decision.outcome == "execute"
    assert decision.command_id == "delete"
    assert decision.agreed is True


def test_strong_fuzzy_alone_executes(monkeypatch):
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=_fuzzy("stop", 90.0)))
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=None))

    decision = cr.resolve_command(
        "නවත්වන්න", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK
    )

    assert decision.outcome == "execute"
    assert decision.command_id == "stop"
    assert decision.agreed is False


def test_strong_embedding_alone_executes(monkeypatch):
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=None, borderline=None))
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=_embedding("next", 0.9)))

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "execute"
    assert decision.command_id == "next"
    assert decision.agreed is False


def test_both_strong_different_commands_asks_for_confirmation(monkeypatch):
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=_fuzzy("next", 88.0)))
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=_embedding("previous", 0.9)))

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "confirm"
    assert decision.command_id is None  # never guess between them
    assert decision.fuzzy_command_id == "next"
    assert decision.embedding_command_id == "previous"


def test_both_weak_with_a_borderline_candidate_asks_for_confirmation(monkeypatch):
    monkeypatch.setattr(
        cr, "match_command", _stub_match_command(strong=None, borderline=_fuzzy("save", 65.0))
    )
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=None, borderline=None))

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "confirm"
    assert decision.command_id is None
    assert decision.fuzzy_command_id == "save"


def test_no_candidate_at_all_is_ordinary_dictation(monkeypatch):
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=None, borderline=None))
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=None, borderline=None))

    decision = cr.resolve_command(
        "අද කාලගුණය හොඳයි", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK
    )

    assert decision.outcome == "none"
    assert decision.command_id is None


def test_destructive_command_below_its_higher_bar_is_not_strong(monkeypatch):
    # Cleared the *normal* embedding threshold but not the destructive one.
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=None, borderline=None))
    monkeypatch.setattr(
        cr,
        "best_match",
        _stub_best_match(strong=_embedding("delete", cr.settings.voice_embedding_destructive_threshold - 0.01)),
    )

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    # Not strong enough to execute on its own, and nothing else supports
    # it either -> falls through to "none", not a guessed execute.
    assert decision.outcome == "none"


def test_destructive_command_can_still_execute_on_one_sufficiently_strong_signal(monkeypatch):
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=_fuzzy("delete", 95.0, destructive=True)))
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=None))

    decision = cr.resolve_command("මකන්න", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "execute"
    assert decision.command_id == "delete"


def test_disabled_flag_falls_back_to_fuzzy_only_behaviour(monkeypatch):
    monkeypatch.setattr(cr.settings, "voice_command_embedding_matching_enabled", False)
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=_fuzzy("stop", 90.0)))
    best_match_calls = []
    monkeypatch.setattr(cr, "best_match", lambda *a, **k: best_match_calls.append(1))

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "execute"
    assert decision.command_id == "stop"
    assert decision.embedding_command_id is None
    assert best_match_calls == []  # embedding path never even consulted


def test_unenrolled_student_empty_bank_falls_back_to_fuzzy_only(monkeypatch):
    monkeypatch.setattr(cr, "match_command", _stub_match_command(strong=None))
    best_match_calls = []
    monkeypatch.setattr(cr, "best_match", lambda *a, **k: best_match_calls.append(1))

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank={})

    assert decision.outcome == "none"
    assert best_match_calls == []
