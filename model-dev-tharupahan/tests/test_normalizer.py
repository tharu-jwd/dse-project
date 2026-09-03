import unicodedata

import pytest

from sinhala_asr.text.normalizer import canonicalize, metric_normalize, transcript_flags


def test_canonicalize_is_conservative_and_idempotent():
    source = "  මම\tසිංහල  කියවමි .  English 123  "
    expected = "මම සිංහල කියවමි. English 123"

    assert canonicalize(source) == expected
    assert canonicalize(expected) == expected


def test_canonicalize_normalizes_unicode_and_removes_invalid_format_chars():
    source = unicodedata.normalize("NFD", "Café සිංහල") + "\u200b"

    assert canonicalize(source) == "Café සිංහල"
    assert "non_nfc" in transcript_flags(source)
    assert "control_or_format_character" in transcript_flags(source)


def test_valid_sinhala_joiner_is_preserved_but_invalid_joiner_is_removed():
    valid = "ක්‍රමය"
    invalid = "ක\u200dම"

    assert canonicalize(valid) == valid
    assert canonicalize(invalid) == "කම"
    assert "invalid_zwj" not in transcript_flags(valid)
    assert "control_or_format_character" not in transcript_flags(valid)
    assert "invalid_zwj" in transcript_flags(invalid)


def test_metric_normalization_removes_punctuation_not_word_spacing():
    assert metric_normalize("ආයුබෝවන්, ලෝකය!") == "ආයුබෝවන් ලෝකය"
    assert metric_normalize("බුදු දහම") != metric_normalize("බුදුදහම")


def test_transcript_flags_code_switching_and_digits():
    flags = transcript_flags("අද meeting එක 10ට")

    assert "code_switched_latin" in flags
    assert "contains_digit" in flags


def test_pure_english_is_not_mislabeled_as_code_switched():
    flags = transcript_flags("English only")
    assert "latin_only" in flags
    assert "code_switched_latin" not in flags


def test_non_string_transcript_is_rejected():
    with pytest.raises(TypeError):
        canonicalize(None)
    assert transcript_flags(None) == ["non_string_transcript"]
