import pytest

from sinhala_asr.training.pilot import select_pilot_rows


def test_pilot_selection_is_deterministic_and_balanced() -> None:
    rows = [
        {
            "sample_id": f"{speaker}-{language}-{index}",
            "speaker_id": speaker,
            "language_class": language,
            "dataset_split": "train",
        }
        for speaker in ("a", "b", "c")
        for language in ("sinhala_only", "latin_only")
        for index in range(4)
    ]
    first = select_pilot_rows(rows, total=12, latin_rows=3, seed=7)
    second = select_pilot_rows(rows, total=12, latin_rows=3, seed=7)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert sum(row["language_class"] == "latin_only" for row in first) == 3
    assert {row["speaker_id"] for row in first} == {"a", "b", "c"}


def test_pilot_selection_rejects_impossible_quota() -> None:
    with pytest.raises(ValueError, match="only selected"):
        select_pilot_rows([], total=1, latin_rows=0, seed=7)
