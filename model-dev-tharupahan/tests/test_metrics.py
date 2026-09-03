from sinhala_asr.evaluation.metrics import edit_counts, evaluate_rows, strict_normalize


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
