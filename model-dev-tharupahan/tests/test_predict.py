import pytest

from sinhala_asr.evaluation.predict import select_prediction_rows


ROWS = [
    {"sample_id": "v", "dataset_split": "validation"},
    {"sample_id": "c", "dataset_split": "validation_candidate"},
    {"sample_id": "t", "dataset_split": "test"},
]


def test_candidate_and_test_guards() -> None:
    with pytest.raises(ValueError, match="unreviewed"):
        select_prediction_rows(ROWS, "validation_candidate")
    assert (
        select_prediction_rows(ROWS, "validation_candidate", allow_unreviewed=True)[0][
            "sample_id"
        ]
        == "c"
    )
    with pytest.raises(ValueError, match="unlock-test"):
        select_prediction_rows(ROWS, "test")
    assert select_prediction_rows(ROWS, "test", unlock_test=True)[0]["sample_id"] == "t"
