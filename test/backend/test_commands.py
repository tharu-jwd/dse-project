import unicodedata

from rapidfuzz import fuzz

from app.streaming.commands import (
    COMMANDS,
    COMMANDS_EN,
    COMMANDS_SI,
    get_commands,
    hotwords_for,
    match_command,
    skeleton,
)


def _command(command_id: str):
    return next(c for c in COMMANDS if c.id == command_id)


def _swap_one_vowel_sign(phrase: str) -> str:
    """Replace the first Sinhala dependent vowel sign/virama in `phrase`
    with a different one, leaving every base consonant untouched.

    Simulates the most common real ASR failure mode for short command
    phrases: Whisper gets the consonants right but guesses the wrong
    vowel sign. Built from the actual phrase rather than a hand-typed
    alternate spelling, so the test does not depend on knowing a second,
    independently-correct Sinhala misspelling.
    """

    alternates = ["ෙ", "ි", "ු", "්"]  # kombuva / is-pilla / paa-pilla / al-lakuna
    chars = list(phrase)

    for index, ch in enumerate(chars):
        if unicodedata.category(ch) in ("Mn", "Mc"):
            replacement = next(alt for alt in alternates if alt != ch)
            chars[index] = replacement
            return "".join(chars)

    raise AssertionError(f"{phrase!r} has no vowel sign to swap")


# --- skeleton() -------------------------------------------------------


def test_skeleton_strips_dependent_vowel_signs():
    phrase = _command("save").phrase
    mutated = _swap_one_vowel_sign(phrase)
    assert mutated != phrase
    assert skeleton(mutated) == skeleton(phrase)


def test_skeleton_ignores_whitespace_and_case():
    submit = _command("submit").phrase
    assert skeleton(submit) == skeleton(submit.replace(" ", "  "))
    assert skeleton("Save") == skeleton("  SAVE  ")


def test_skeleton_is_stable_for_command_vocabulary():
    # Every command phrase must produce a non-empty skeleton, otherwise it
    # could never be matched against.
    for command in COMMANDS:
        assert skeleton(command.phrase) != ""


# --- match_command() ---------------------------------------------------


def test_exact_match_returns_the_command_with_a_high_score():
    result = match_command(_command("save").phrase)
    assert result is not None
    assert result.command.id == "save"
    assert result.score >= 95


def test_near_miss_vowel_spelling_still_matches():
    near_miss = _swap_one_vowel_sign(_command("submit").phrase)
    result = match_command(near_miss)
    assert result is not None
    assert result.command.id == "submit"


def test_unrelated_text_does_not_match():
    result = match_command("අද කාලගුණය හොඳයි")
    assert result is None


def _partial_skeleton_score(command_id: str) -> tuple[str, float]:
    """A text whose skeleton is `command_id`'s phrase with its last
    character dropped, plus the score rapidfuzz gives it against that
    command - computed directly rather than assumed, since skeleton()
    collapses vowel-sign-only differences to a 100% match.

    Every phrase now shares the same leading wake word, so truncating
    from the *front* (or taking a short prefix) would produce text that
    fuzzy-matches several short commands' shared "zimi" prefix almost
    equally well. Dropping one character from the *end* keeps the whole
    distinguishing tail, which keeps this command the clear winner
    against the rest of the vocabulary - asserted below rather than
    assumed, since match_command always picks the vocabulary-wide best
    score, not just this one command's score.
    """

    full_skeleton = skeleton(_command(command_id).phrase)
    partial_text = full_skeleton[:-1]
    scores = {c.id: fuzz.ratio(skeleton(partial_text), skeleton(c.phrase)) for c in COMMANDS}
    score = scores[command_id]
    assert 0 < score < 100
    assert score == max(scores.values()), (
        f"{command_id}'s truncated phrase no longer wins the fuzzy match: {scores}"
    )
    return partial_text, score


def test_below_threshold_returns_none_instead_of_a_guess():
    partial_text, score = _partial_skeleton_score("save")
    result = match_command(partial_text, threshold=score + 1)
    assert result is None


def test_destructive_commands_need_a_higher_threshold():
    # A score that clears the normal threshold but not the stricter
    # destructive one.
    partial_text, score = _partial_skeleton_score("delete")
    result = match_command(partial_text, threshold=score - 1, destructive_threshold=score + 1)
    assert result is None


def test_low_asr_confidence_rejects_a_borderline_match():
    # Borderline fuzzy score + low avg_logprob => no match.
    partial_text, score = _partial_skeleton_score("delete")
    result = match_command(
        partial_text,
        threshold=score - 1,
        destructive_threshold=score - 1,
        avg_logprob=-2.0,
        logprob_floor=-0.5,
    )
    assert result is None


def test_low_asr_confidence_does_not_reject_a_strong_match():
    result = match_command(
        _command("save").phrase,
        avg_logprob=-2.0,
        logprob_floor=-0.5,
    )
    assert result is not None
    assert result.command.id == "save"


def test_empty_transcript_does_not_match():
    assert match_command("") is None
    assert match_command("   ") is None


# --- Two languages, chosen without touching the fuzzy-matching logic ---


def test_commands_default_is_the_sinhala_set():
    assert COMMANDS is COMMANDS_SI


def test_get_commands_returns_the_right_set_per_language():
    assert get_commands("si") == COMMANDS_SI
    assert get_commands("en") == COMMANDS_EN


def test_get_commands_falls_back_to_sinhala_for_an_unknown_language():
    assert get_commands("fr") == COMMANDS_SI


def test_same_ids_exist_in_both_languages():
    """resolve_command, the frontend's command mapping and
    _ACTIONABLE_NOTE_COMMANDS all key off the id, never the phrase - the
    two phrase sets must stay id-for-id identical."""

    assert {c.id for c in COMMANDS_SI} == {c.id for c in COMMANDS_EN}


def test_destructive_flag_matches_across_languages():
    destructive_si = {c.id for c in COMMANDS_SI if c.destructive}
    destructive_en = {c.id for c in COMMANDS_EN if c.destructive}
    assert destructive_si == destructive_en == {"submit", "delete"}


def test_match_command_defaults_to_sinhala():
    result = match_command(_command("next").phrase)
    assert result is not None
    assert result.command.id == "next"
    # The same text is meaningless in the English set - no accidental match.
    assert match_command(_command("next").phrase, language="en") is None


def test_match_command_can_match_english_phrases():
    english_next = next(c for c in COMMANDS_EN if c.id == "next").phrase
    result = match_command(english_next, language="en")
    assert result is not None
    assert result.command.id == "next"
    # And the English phrase doesn't fuzzy-match anything in Sinhala either.
    assert match_command(english_next, language="si") is None


def test_hotwords_for_returns_the_right_language():
    assert hotwords_for("si") == " ".join(c.phrase for c in COMMANDS_SI)
    assert hotwords_for("en") == " ".join(c.phrase for c in COMMANDS_EN)
    assert "next" in hotwords_for("en")
    assert "next" not in hotwords_for("si")
