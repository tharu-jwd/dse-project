from sinhala_asr.data.splits import make_speaker_disjoint_candidates


def test_candidate_splits_are_speaker_disjoint_and_invalid_rows_are_explicit() -> None:
    rows = [
        {"sample_id": f"{speaker}-{index}", "speaker_key": speaker, "is_valid": True}
        for speaker in ["a", "b", "c", "d"]
        for index in range(3)
    ]
    rows.append({"sample_id": "bad", "speaker_key": "z", "is_valid": False})
    assigned, summary = make_speaker_disjoint_candidates(rows, candidate_rows=2, seed=7)

    by_sample = {row["sample_id"]: row for row in assigned}
    validation = {
        row["speaker_key"]
        for row in assigned
        if row["dataset_split"] == "validation_candidate"
    }
    test = {
        row["speaker_key"]
        for row in assigned
        if row["dataset_split"] == "test_candidate"
    }
    train = {row["speaker_key"] for row in assigned if row["dataset_split"] == "train"}

    assert validation and test
    assert not (validation & test or validation & train or test & train)
    assert by_sample["bad"]["dataset_split"] == "excluded"
    assert by_sample["bad"]["exclusion_reason"] == "automatic_validation_failure"
    assert summary["speaker_disjoint"] is True
