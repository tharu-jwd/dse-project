from sinhala_asr.review.suggestions import classify_change


def test_classifies_suggestion_changes() -> None:
    assert classify_change("මම යමි", "මම යමි") == "unchanged"
    assert classify_change("මම යමි.", "මම යමි") == "format_case_punctuation_only"
    assert classify_change("තේරුම් ගන්න", "තේරුම්ගන්න") == "spacing_only"
    assert classify_change("නිවරදි", "නිවැරදි") == "lexical_or_spelling"
