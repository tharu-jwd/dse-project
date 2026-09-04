from sinhala_asr.evaluation.metrics import (
    edit_counts,
    evaluate_rows,
    paired_delta_interval,
    strict_normalize,
)


def test_edit_counts_distinguish_operations() -> None:
    assert edit_counts(["a", "b"], ["a", "c", "d"]) == {
        "errors": 2,
        "substitutions": 1,
        "deletions": 0,
        "insertions": 1,
        "reference_units": 2,
    }


def test_evaluation_reports_strict_canonical_and_subgroups() -> None:
    rows = [
        {
            "sample_id": "1",
            "reference": "මම යමි.",
            "prediction": "මම යමි",
            "language_class": "sinhala_only",
        },
        {
            "sample_id": "2",
            "reference": "hello ලෝකය",
            "prediction": "hello ලෝකය",
            "language_class": "code_switched",
        },
    ]
    scored, summary = evaluate_rows(rows, bootstrap_iterations=20)
    assert summary["strict"]["wer"] > summary["canonical"]["wer"]
    assert summary["canonical"]["wer"] == 0
    assert summary["by_language_class"]["code_switched"]["strict"]["wer"] == 0
    assert "normalization_only" in scored[0]["error_labels"]


def test_strict_normalization_only_collapses_whitespace_and_nfc() -> None:
    assert strict_normalize("  මම\n  යමි. ") == "මම යමි."


def test_paired_delta_interval_detects_consistent_improvement() -> None:
    baseline, _ = evaluate_rows(
        [
            {"reference": "one two", "prediction": "wrong wrong"},
            {"reference": "three four", "prediction": "wrong wrong"},
        ],
        bootstrap_iterations=0,
    )
    candidate, _ = evaluate_rows(
        [
            {"reference": "one two", "prediction": "one two"},
            {"reference": "three four", "prediction": "three four"},
        ],
        bootstrap_iterations=0,
    )
    interval = paired_delta_interval(
        baseline, candidate, "strict", iterations=20, seed=7
    )
    assert interval["wer"] == [-1.0, -1.0]
    assert interval["cer"][1] < 0
