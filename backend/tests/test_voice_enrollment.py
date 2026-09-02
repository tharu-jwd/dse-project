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


# --- Two languages, kept independent without any code change ------------


def test_sinhala_and_english_samples_for_the_same_id_never_collide(db_user):
    """Recording "delete" in both languages must produce two separate
    banks, not one clobbering the other - same command id, same user,
    different language."""

    voice_enrollment.submit_sample(db_user, "delete", _vector(1.0, 0.0, 0.0), language="si")
    voice_enrollment.submit_sample(db_user, "delete", _vector(0.0, 1.0, 0.0), language="en")

    si_bank = voice_enrollment.load_bank(db_user, language="si")
    en_bank = voice_enrollment.load_bank(db_user, language="en")

    assert len(si_bank["delete"]) == 1
    assert len(en_bank["delete"]) == 1
    assert not np.array_equal(si_bank["delete"][0], en_bank["delete"][0])


def test_progress_is_scoped_to_one_language(db_user):
    voice_enrollment.submit_sample(db_user, "save", _vector(1.0, 0.0), language="si")

    si_progress = next(p for p in voice_enrollment.get_progress(db_user, "si") if p.command_id == "save")
    en_progress = next(p for p in voice_enrollment.get_progress(db_user, "en") if p.command_id == "save")

    assert si_progress.collected == 1
    assert en_progress.collected == 0


def test_deleting_one_languages_samples_leaves_the_other_untouched(db_user):
    """The independence the whole feature is built for: delete + re-record
    for one language must never touch the other language's enrollment,
    and neither operation requires editing commands.py or any code."""

    voice_enrollment.submit_sample(db_user, "next", _vector(1.0, 0.0), language="si")
    voice_enrollment.submit_sample(db_user, "next", _vector(0.0, 1.0), language="en")

    voice_enrollment.delete_samples(db_user, "next", language="en")

    si_progress = next(p for p in voice_enrollment.get_progress(db_user, "si") if p.command_id == "next")
    en_progress = next(p for p in voice_enrollment.get_progress(db_user, "en") if p.command_id == "next")
    assert si_progress.collected == 1
    assert en_progress.collected == 0


def test_unknown_command_id_for_a_language_is_rejected(db_user):
    # "delete" is valid, but only for si/en - an unrecognized language
    # code has no commands at all to validate against.
    with pytest.raises(voice_enrollment.UnknownCommandError):
        voice_enrollment.submit_sample(db_user, "delete", _vector(1.0, 0.0), language="fr")


# --- Practice / compare -----------------------------------------------


def test_practice_reports_similarity_against_own_enrolled_samples(db_user):
    voice_enrollment.submit_sample(db_user, "stop", _vector(1.0, 0.0, 0.0))
    voice_enrollment.submit_sample(db_user, "stop", _vector(0.99, 0.01, 0.0))

    result = voice_enrollment.practice_command(db_user, "stop", _vector(0.98, 0.02, 0.0))

    assert result.command_id == "stop"
    assert result.enrolled_sample_count == 2
    assert result.own_similarity == pytest.approx(
        max(
            manhattan_similarity(_vector(0.98, 0.02, 0.0), _vector(1.0, 0.0, 0.0)),
            manhattan_similarity(_vector(0.98, 0.02, 0.0), _vector(0.99, 0.01, 0.0)),
        ),
        abs=1e-4,
    )
    assert result.passes_threshold == (
        result.own_similarity >= voice_enrollment.settings.voice_embedding_similarity_threshold
    )


def test_practice_with_no_enrolled_samples_reports_nothing_to_compare(db_user):
    result = voice_enrollment.practice_command(db_user, "submit", _vector(1.0, 0.0))

    assert result.enrolled_sample_count == 0
    assert result.own_similarity is None
    assert result.passes_threshold is False


def test_practice_flags_a_closer_match_to_a_different_enrolled_command(db_user):
    """The whole point of the comparison tool: catch an attempt that
    actually sounds more like a *different* enrolled command than the
    one it was meant for."""

    voice_enrollment.submit_sample(db_user, "next", _vector(1.0, 0.0, 0.0))
    voice_enrollment.submit_sample(db_user, "previous", _vector(0.0, 1.0, 0.0))

    # Attempted as "next" but sounds exactly like the enrolled "previous".
    result = voice_enrollment.practice_command(db_user, "next", _vector(0.0, 1.0, 0.0))

    assert result.closest_other_command_id == "previous"
    assert result.closest_other_similarity == pytest.approx(1.0, abs=1e-4)


def test_practice_rejects_unknown_command_id(db_user):
    with pytest.raises(voice_enrollment.UnknownCommandError):
        voice_enrollment.practice_command(db_user, "not_a_real_command", _vector(1.0, 0.0))
