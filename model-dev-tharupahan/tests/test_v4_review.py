import pytest

from sinhala_asr.review.v4 import finalize_v4_rows, select_v4_review_rows


def row(sample_id: str, split: str, original: str, canonical: str) -> dict:
    return {
        "sample_id": sample_id,
        "dataset_split": split,
        "text_original": original,
        "text_canonical": canonical,
        "text_metric": canonical,
    }


def test_selection_includes_all_disputed_and_fixed_controls() -> None:
    rows = [
        row("changed", "validation", "helo", "hello"),
        row("same-a", "validation", "same", "same"),
        row("same-b", "test", "same", "same"),
        row("train", "train", "helo", "hello"),
    ]
    selected = select_v4_review_rows(rows, control_rows=1, seed=3)
    assert {item["sample_id"] for item in selected} >= {"changed"}
    assert len(selected) == 2
    changed = next(item for item in selected if item["sample_id"] == "changed")
    assert changed["text_original"] == "helo"
    assert changed["previous_v3_transcript"] == "hello"


def test_finalizer_restores_original_then_applies_audio_decisions() -> None:
    rows = [
        row("keep-original", "validation", "helo", "hello"),
        row("verified-edit", "test", "wrld", "world"),
    ]
    finalized, summary = finalize_v4_rows(
        rows,
        {"keep-original", "verified-edit"},
        {
            "keep-original": {"decision": "correct"},
            "verified-edit": {"decision": "edited", "text_corrected": "වර්ල්ඩ්"},
        },
    )
    assert finalized[0]["text_canonical"] == "helo"
    assert finalized[1]["text_canonical"] == "වර්ල්ඩ්"
    assert summary["audio_verified_edits"] == 1


def test_finalizer_requires_complete_review() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        finalize_v4_rows(
            [row("a", "validation", "a", "a")], {"a"}, {}
        )


def test_finalizer_holds_out_unreviewed_evaluation_rows() -> None:
    finalized, summary = finalize_v4_rows(
        [
            row("reviewed", "validation", "a", "a"),
            row("unreviewed", "test", "b", "b"),
        ],
        {"reviewed"},
        {"reviewed": {"decision": "correct"}},
    )
    assert finalized[0]["dataset_split"] == "validation"
    assert finalized[1]["dataset_split"] == "heldout_unreviewed"
    assert summary["heldout_unreviewed_rows"] == 1
