from scripts.refinement.build_label_refinement_ab import build_arms


def row(sample_id: str, split: str, language: str, text: str) -> dict:
    return {
        "sample_id": sample_id,
        "dataset_split": split,
        "language_class": language,
        "text_canonical": text,
        "text_metric": text,
    }


def test_builds_matched_arms_and_never_changes_validation() -> None:
    manifest = [
        row("si-train", "train", "sinhala_only", "හොදයි"),
        row("en-train", "train", "latin_only", "helo"),
        row("si-val", "validation", "sinhala_only", "වලංගු"),
        row("en-val", "validation", "latin_only", "valid"),
    ]
    selected = [
        {"sample_id": "si-train", "language_class": "sinhala_only", "original": "හොදයි"},
        {"sample_id": "en-train", "language_class": "latin_only", "original": "helo"},
    ]
    refinements = [
        {"sample_id": "si-train", "original": "හොදයි", "corrected": "හොඳයි", "changed": True, "confidence": "high"},
        {"sample_id": "en-train", "original": "helo", "corrected": "hello", "changed": True, "confidence": "medium"},
    ]
    arms = build_arms(
        manifest,
        selected,
        refinements,
        accepted_confidence={"high"},
        validation_limits={"sinhala_only": 1, "latin_only": 1},
    )
    assert arms["sinhala_only"]["refined"][0]["text_canonical"] == "හොඳයි"
    assert arms["latin_only"]["refined"][0]["text_canonical"] == "helo"
    assert arms["sinhala_only"]["original"][1] == arms["sinhala_only"]["refined"][1]
