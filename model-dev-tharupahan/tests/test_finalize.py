import pytest

from sinhala_asr.review.finalize import finalize_rows


def candidates():
    return [
        {
            "sample_id": "train",
            "dataset_split": "train",
            "text_original": "a",
            "text_canonical": "a",
        },
        {
            "sample_id": "val",
            "dataset_split": "validation_candidate",
            "text_original": "වැරදි",
            "text_canonical": "වැරදි",
        },
        {
            "sample_id": "test",
            "dataset_split": "test_candidate",
            "text_original": "හරි",
            "text_canonical": "හරි",
        },
    ]


def test_finalize_requires_every_candidate() -> None:
    with pytest.raises(ValueError, match="1 gold candidates"):
        finalize_rows(
            candidates(), {"val": {"decision": "correct", "text_corrected": "වැරදි"}}
        )


def test_finalize_applies_corrections_and_exclusions() -> None:
    rows, summary = finalize_rows(
        candidates(),
        {
            "val": {"decision": "edited", "text_corrected": "නිවැරදි"},
            "test": {"decision": "bad_audio", "text_corrected": "හරි"},
        },
    )
    by_id = {row["sample_id"]: row for row in rows}
    assert by_id["val"]["dataset_split"] == "validation"
    assert by_id["val"]["text_canonical"] == "නිවැරදි"
    assert by_id["test"]["dataset_split"] == "excluded"
    assert by_id["test"]["exclusion_reason"] == "native_review_bad_audio"
    assert summary["adjudicated_candidates"] == 2
