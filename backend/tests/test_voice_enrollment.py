"""Tests for app.services.voice_enrollment against the real dev database
(see tests/conftest.py - there's no isolated test-DB setup in this
project yet). Each test uses its own throwaway user so rows never
collide, and command_enrollments rows are cleaned up via cascade delete
when the fixture removes the user.
"""

import numpy as np
import pytest

from app.services import voice_enrollment
from app.streaming.commands import COMMANDS
from app.streaming.embeddings import l2_normalize, manhattan_similarity


def _vector(*values: float) -> np.ndarray:
    return l2_normalize(np.array(values, dtype=np.float32))


def test_progress_starts_at_zero_for_every_command(db_user):
    progress = voice_enrollment.get_progress(db_user)
    assert {item.command_id for item in progress} == {c.id for c in COMMANDS}
    assert all(item.collected == 0 and not item.complete for item in progress)


def test_first_sample_for_a_command_is_always_accepted(db_user):
    # Nothing to compare it against yet - can't reject with no baseline.
    result = voice_enrollment.submit_sample(db_user, "stop", _vector(1.0, 0.0, 0.0))
    assert result.accepted
    assert result.similarity is None
    assert result.collected == 1


def test_similar_second_sample_is_accepted_and_counted(db_user):
    voice_enrollment.submit_sample(db_user, "stop", _vector(1.0, 0.0, 0.0))
    result = voice_enrollment.submit_sample(db_user, "stop", _vector(0.99, 0.01, 0.0))
    assert result.accepted
    # Enrollment gating scores on Manhattan similarity too, same as
    # runtime matching (app.streaming.embeddings.manhattan_similarity) -
    # not the raw cosine dot product.
    assert result.similarity == pytest.approx(
        manhattan_similarity(_vector(1.0, 0.0, 0.0), _vector(0.99, 0.01, 0.0)), abs=1e-4
    )
    assert result.collected == 2


def test_outlier_sample_is_rejected_not_stored(db_user):
    voice_enrollment.submit_sample(db_user, "stop", _vector(1.0, 0.0, 0.0))
    # Wildly different direction from the first accepted "stop" sample -
    # simulates a mis-recording (wrong word, false start, background noise).
    result = voice_enrollment.submit_sample(db_user, "stop", _vector(0.0, 1.0, 0.0))

    assert not result.accepted
    assert result.reason == "low_similarity"
    assert result.similarity < voice_enrollment.settings.voice_enrollment_min_sample_similarity
    # Rejected sample must not count toward progress.
    assert result.collected == 1
    progress = next(p for p in voice_enrollment.get_progress(db_user) if p.command_id == "stop")
    assert progress.collected == 1


def test_cannot_exceed_required_sample_count(db_user):
    required = voice_enrollment.settings.voice_enrollment_samples_required
    base = _vector(1.0, 0.0, 0.0)

    for _ in range(required):
        result = voice_enrollment.submit_sample(db_user, "save", base)
        assert result.accepted

    overflow = voice_enrollment.submit_sample(db_user, "save", base)
    assert not overflow.accepted
    assert overflow.reason == "already_complete"
    assert overflow.collected == required


def test_delete_samples_resets_progress(db_user):
    voice_enrollment.submit_sample(db_user, "next", _vector(1.0, 0.0))
    voice_enrollment.submit_sample(db_user, "next", _vector(0.99, 0.01))
    voice_enrollment.delete_samples(db_user, "next")

    progress = next(p for p in voice_enrollment.get_progress(db_user) if p.command_id == "next")
    assert progress.collected == 0


def test_unknown_command_id_is_rejected(db_user):
    with pytest.raises(voice_enrollment.UnknownCommandError):
        voice_enrollment.submit_sample(db_user, "not_a_real_command", _vector(1.0, 0.0))

    with pytest.raises(voice_enrollment.UnknownCommandError):
        voice_enrollment.delete_samples(db_user, "not_a_real_command")


def test_load_bank_groups_by_command_for_matching(db_user):
    voice_enrollment.submit_sample(db_user, "delete", _vector(1.0, 0.0, 0.0))
    voice_enrollment.submit_sample(db_user, "delete", _vector(0.99, 0.01, 0.0))
    voice_enrollment.submit_sample(db_user, "stop", _vector(0.0, 1.0, 0.0))

    bank = voice_enrollment.load_bank(db_user)

    assert set(bank.keys()) == {"delete", "stop"}
    assert len(bank["delete"]) == 2
    assert len(bank["stop"]) == 1


def test_load_bank_skips_samples_from_a_different_model_version(db_user, monkeypatch):
    voice_enrollment.submit_sample(db_user, "delete", _vector(1.0, 0.0))

    monkeypatch.setattr(voice_enrollment.settings, "voice_embedding_model_version", "some-other-checkpoint")
    bank = voice_enrollment.load_bank(db_user)

    assert bank == {}
