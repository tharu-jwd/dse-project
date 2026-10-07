"""Tests for resolve_command's embedding-only decision logic - strong match,
the destructive-command bar, borderline confirmation, and the no-bank case.
`best_match` is stubbed out here since its own matching behaviour is covered
by test_embeddings.py; this file is purely about `resolve_command`'s decision
logic given controlled inputs.

Commands are identified from embeddings only - the old fuzzy-text path is
commented out in command_resolution.py, so the transcript passed in is never
used to decide anything.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from app.streaming import command_resolution as cr


def _embedding(label: str, score: float):
    return SimpleNamespace(label=label, score=score)


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


def test_strong_embedding_executes(monkeypatch):
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=_embedding("next", 0.9)))

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "execute"
    assert decision.command_id == "next"
    assert decision.embedding_command_id == "next"
    assert decision.fuzzy_command_id is None
    assert decision.agreed is False


def test_the_transcript_never_decides_anything(monkeypatch):
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=None, borderline=None))

    # A transcript that is exactly a command phrase must not execute it.
    decision = cr.resolve_command("delete", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "none"
    assert decision.command_id is None


def test_borderline_embedding_asks_for_confirmation(monkeypatch):
    monkeypatch.setattr(
        cr, "best_match", _stub_best_match(strong=None, borderline=_embedding("save", 0.78))
    )

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "confirm"
    assert decision.command_id is None
    assert decision.embedding_command_id == "save"


def test_no_candidate_at_all_is_ordinary_dictation(monkeypatch):
    monkeypatch.setattr(cr, "best_match", _stub_best_match(strong=None, borderline=None))

    decision = cr.resolve_command(
        "අද කාලගුණය හොඳයි", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK
    )

    assert decision.outcome == "none"
    assert decision.command_id is None


def test_destructive_command_below_its_higher_bar_is_not_strong(monkeypatch):
    # Cleared the *normal* embedding threshold but not the destructive one.
    monkeypatch.setattr(
        cr,
        "best_match",
        _stub_best_match(strong=_embedding("delete", cr.settings.voice_embedding_destructive_threshold - 0.01)),
    )

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    # Not strong enough to execute, and nothing else supports it -> never a guessed execute.
    assert decision.outcome != "execute"
    assert decision.command_id is None


def test_destructive_command_executes_above_its_higher_bar(monkeypatch):
    monkeypatch.setattr(
        cr,
        "best_match",
        _stub_best_match(strong=_embedding("delete", cr.settings.voice_embedding_destructive_threshold + 0.01)),
    )

    decision = cr.resolve_command("...", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "execute"
    assert decision.command_id == "delete"


def test_disabled_flag_identifies_nothing(monkeypatch):
    monkeypatch.setattr(cr.settings, "voice_command_embedding_matching_enabled", False)
    best_match_calls = []
    monkeypatch.setattr(cr, "best_match", lambda *a, **k: best_match_calls.append(1))

    decision = cr.resolve_command("delete", avg_logprob=-0.1, embedding=np.zeros(4), bank=NON_EMPTY_BANK)

    assert decision.outcome == "none"  # no text fallback any more
    assert best_match_calls == []


def test_unenrolled_student_empty_bank_identifies_nothing(monkeypatch):
    best_match_calls = []
    monkeypatch.setattr(cr, "best_match", lambda *a, **k: best_match_calls.append(1))

    decision = cr.resolve_command("delete", avg_logprob=-0.1, embedding=np.zeros(4), bank={})

    assert decision.outcome == "none"
    assert best_match_calls == []
