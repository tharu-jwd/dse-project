from scripts.data.prepare_librispeech_replay_colab import speaker_balanced_ids


def test_speaker_balanced_ids_is_deterministic_and_spreads_speakers():
    rows = [
        {"id": f"a-{number}", "speaker_id": "a"} for number in range(3)
    ] + [{"id": f"b-{number}", "speaker_id": "b"} for number in range(3)]

    selected = speaker_balanced_ids(rows, 4, 17)

    assert selected == speaker_balanced_ids(list(reversed(rows)), 4, 17)
    assert {sample.split("-")[0] for sample in selected[:2]} == {"a", "b"}
    assert len(selected) == len(set(selected)) == 4


def test_speaker_balanced_ids_rejects_oversized_request():
    rows = [{"id": "only", "speaker_id": "speaker"}]

    try:
        speaker_balanced_ids(rows, 2, 17)
    except ValueError as error:
        assert "requested 2 rows" in str(error)
    else:
        raise AssertionError("oversized selection should fail")
