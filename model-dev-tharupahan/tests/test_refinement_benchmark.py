import pandas as pd

from scripts.evaluation.export_verified_refinement_benchmark import build_benchmark_rows
from scripts.evaluation.score_refinement_benchmark import score_benchmark


def test_export_excludes_bad_audio_and_controls() -> None:
    queue = pd.DataFrame(
        [
            {"sample_id": "a", "review_category": "disputed_validation", "text_original": "helo", "language_class": "latin_only"},
            {"sample_id": "b", "review_category": "disputed_test", "text_original": "bad", "language_class": "latin_only"},
            {"sample_id": "c", "review_category": "control_test", "text_original": "same", "language_class": "latin_only"},
        ]
    )
    inputs, truth = build_benchmark_rows(
        queue,
        {
            "a": {"decision": "edited", "text_corrected": "hello"},
            "b": {"decision": "bad_audio"},
        },
    )
    assert [row["sample_id"] for row in inputs] == ["a"]
    assert truth[0]["verified"] == "hello"


def test_score_reports_exact_and_false_rewrites() -> None:
    truth = [
        {"sample_id": "a", "original": "helo", "verified": "hello"},
        {"sample_id": "b", "original": "same", "verified": "same"},
    ]
    predictions = [
        {"sample_id": "a", "corrected": "hello"},
        {"sample_id": "b", "corrected": "changed"},
    ]
    score = score_benchmark(truth, predictions)
    assert score["confusion"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 0}
    assert score["exact_matches"] == 1
    assert score["change_precision"] == 0.5
    assert score["exact_proposal_precision"] == 0.5
    assert score["rows_improved_by_character_distance"] == 1
    assert score["rows_worsened_by_character_distance"] == 1
