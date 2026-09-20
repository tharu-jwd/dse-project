"""Adversarial safety tests for destructive voice commands (submit,
delete). A false match here is not a UX annoyance - it destroys a
student's work - so this asks a different question than
test_commands.py's ordinary matching tests: not "does the real phrase
match", but "do near-miss phrases reliably NOT match as destructive".

match_command() already refuses to fall back to a weaker command when
the best candidate doesn't clear its own threshold (see commands.py) -
a destructive candidate below `voice_command_destructive_threshold`
yields None, not a downgraded non-destructive guess. So the bar here is
simple and absolute: none of the near-misses below may return a
destructive CommandMatch.

Wake-word gating (whether a destructive command is even reachable
without "zimi" having been heard) is a separate mechanism
(split_wake_prefix + the streaming route's wake window) - tested here at
the unit level; the full route-level gate is covered by
test_streaming_commands_route.py.
"""

import pytest

from app.streaming.commands import match_command, split_wake_prefix


# (language, phrase, description) - near neighbours, shared prefixes,
# partial words, and the destructive phrase spoken without the wake word.
# Sinhala destructive phrases: submit = "ඉදිරිපත් කරන්න", delete = "මකන්න".
NEAR_MISSES = [
    ("si", "ඉදිරිපත්", "submit, missing the second word"),
    ("si", "කරන්න", "submit, only the second word"),
    ("si", "ඉදිරි", "submit, truncated mid-word"),
    ("si", "මක", "delete, truncated mid-word"),
    ("si", "මකනවා", "delete, different verb conjugation"),
    ("si", "සුරකින්න", "save - shares a suffix with delete, must not cross-match destructive"),
    ("si", "පිළිතුර කරන්න", "answer + karanna - shares a word with submit"),
    ("si", "නවත්වන්න", "stop - unrelated, similar length"),
    ("si", "ආපසු", "previous - unrelated distractor"),
    ("en", "summit", "submit, one-letter slip"),
    ("en", "sub", "submit, truncated"),
    ("en", "submitted", "submit, wrong tense"),
    ("en", "submitting", "submit, wrong tense"),
    ("en", "delta", "delete, near neighbour"),
    ("en", "dell", "delete, truncated + slip"),
    ("en", "deleted", "delete, wrong tense"),
    ("en", "elite", "delete, anagram-ish near neighbour"),
    ("en", "let it", "delete, shares letters, different word boundary"),
    ("en", "commit", "submit, near neighbour (shares 'mit')"),
    ("en", "sunset", "submit, unrelated similar length"),
    ("en", "cancel", "unrelated command word, distractor"),
]


@pytest.mark.parametrize("language,phrase,description", NEAR_MISSES, ids=[d for _, _, d in NEAR_MISSES])
def test_near_miss_never_resolves_to_a_destructive_command(language, phrase, description):
    match = match_command(phrase, language=language)

    if match is not None:
        assert not match.command.destructive, (
            f"{description!r} ({phrase!r}, {language}) falsely matched destructive "
            f"command {match.command.id!r} at score {match.score:.1f} - "
            "this would destroy a student's work on a near-miss."
        )


def test_near_miss_scores_stay_meaningfully_below_the_destructive_bar(capsys):
    """Not just pass/fail - print the actual margin so a shrinking gap is
    visible before it becomes a real false match."""

    from app.core.config import settings
    from app.streaming.commands import get_commands, skeleton
    from rapidfuzz import fuzz

    print(f"\n{'phrase':<20} {'lang':<4} {'closest destructive':<22} {'score':>6} {'bar':>6}")
    worst_margin = 100.0
    for language, phrase, _ in NEAR_MISSES:
        query = skeleton(phrase)
        destructive_commands = [c for c in get_commands(language) if c.destructive]
        best_id, best_score = None, 0.0
        for command in destructive_commands:
            score = fuzz.ratio(query, skeleton(command.phrase))
            if score > best_score:
                best_score, best_id = score, command.id
        bar = settings.voice_command_destructive_threshold
        margin = bar - best_score
        worst_margin = min(worst_margin, margin)
        print(f"{phrase:<20} {language:<4} {best_id or '-':<22} {best_score:>6.1f} {bar:>6.1f}")

    assert worst_margin > 0, "a near-miss scored at or above the destructive threshold"


def test_a_vowel_sign_slip_on_the_same_word_still_matches_by_design():
    """This is NOT a near-miss in the safety sense above: skeleton()
    deliberately strips Sinhala vowel signs so that Whisper mis-guessing
    one on an otherwise-correct word (a common, documented ASR error -
    see skeleton()'s docstring) still matches. "මකින්න" collapses to the
    same consonant skeleton as "මකන්න" ("delete") and is expected to
    match at 100 - the opposite of a false positive, this is the
    tolerance the function exists to provide."""

    match = match_command("මකින්න", language="si")
    assert match is not None
    assert match.command.id == "delete"


def test_a_confirmed_destructive_command_still_works():
    """The guard must not simply block everything - the real phrase,
    spoken cleanly, still executes."""

    submit = match_command("ඉදිරිපත් කරන්න", language="si")
    assert submit is not None
    assert submit.command.id == "submit"

    delete_en = match_command("delete", language="en")
    assert delete_en is not None
    assert delete_en.command.id == "delete"


# --- Wake-word gating -----------------------------------------------


def test_wake_word_is_stripped_when_present():
    woke, rest = split_wake_prefix("zimi delete")
    assert woke is True
    assert rest.strip() == "delete"


def test_same_phrase_without_the_wake_word_is_not_treated_as_woken():
    woke, rest = split_wake_prefix("delete")
    assert woke is False
    assert rest.strip() == "delete"


def test_wake_word_variants_are_all_recognised():
    for variant in ("zimi", "zini", "සිමි"):
        woke, _ = split_wake_prefix(f"{variant} submit")
        assert woke is True, f"wake variant {variant!r} was not recognised"


def test_the_command_word_alone_never_doubles_as_the_wake_word():
    """Saying "delete" by itself must never be interpreted as
    wake-word-plus-empty-command - split_wake_prefix must not match the
    wake word against an unrelated first word."""

    woke, rest = split_wake_prefix("delete everything")
    assert woke is False
    assert rest.strip() == "delete everything"
