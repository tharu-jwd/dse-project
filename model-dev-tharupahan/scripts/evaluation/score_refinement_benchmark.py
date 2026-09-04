#!/usr/bin/env python3
"""Score text-refinement suggestions against native audio-verified labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sinhala_asr.evaluation.metrics import score_pair, strict_normalize


def score_benchmark(truth: list[dict], predictions: list[dict]) -> dict:
    predicted = {str(row["sample_id"]): row for row in predictions}
    if len(predicted) != len(predictions):
        raise ValueError("predictions contain duplicate sample IDs")
    expected_ids = {str(row["sample_id"]) for row in truth}
    if set(predicted) != expected_ids:
        raise ValueError("truth and prediction sample IDs differ")

    tp = fp = fn = tn = exact = 0
    improved = worsened = tied = 0
    exact_changed_proposals = changed_proposals = 0
    model_word_errors = model_word_units = 0
    original_word_errors = original_word_units = 0
    model_char_errors = model_char_units = 0
    original_char_errors = original_char_units = 0
    rows = []
    for target in truth:
        sample_id = str(target["sample_id"])
        result = predicted[sample_id]
        original = str(target["original"])
        verified = str(target["verified"])
        proposed = str(result["corrected"])
        should_change = verified != original
        did_change = proposed != original
        exact_match = strict_normalize(proposed) == strict_normalize(verified)
        if should_change and did_change:
            tp += 1
        elif not should_change and did_change:
            fp += 1
        elif should_change:
            fn += 1
        else:
            tn += 1
        exact += int(exact_match)
        model_score = score_pair(verified, proposed, strict_normalize)
        original_score = score_pair(verified, original, strict_normalize)
        model_word_errors += model_score["word_errors"]
        model_word_units += model_score["word_reference_units"]
        model_char_errors += model_score["character_errors"]
        model_char_units += model_score["character_reference_units"]
        original_word_errors += original_score["word_errors"]
        original_word_units += original_score["word_reference_units"]
        original_char_errors += original_score["character_errors"]
        original_char_units += original_score["character_reference_units"]
        changed_proposals += int(did_change)
        exact_changed_proposals += int(did_change and exact_match)
        if model_score["character_errors"] < original_score["character_errors"]:
            improved += 1
        elif model_score["character_errors"] > original_score["character_errors"]:
            worsened += 1
        else:
            tied += 1
        rows.append(
            {
                "sample_id": sample_id,
                "original": original,
                "verified": verified,
                "proposed": proposed,
                "should_change": should_change,
                "did_change": did_change,
                "exact_match": exact_match,
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
            }
        )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "rows": len(truth),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "change_precision": precision,
        "change_recall": recall,
        "exact_matches": exact,
        "exact_match_rate": exact / len(truth) if truth else 0.0,
        "changed_proposals": changed_proposals,
        "exact_changed_proposals": exact_changed_proposals,
        "exact_proposal_precision": (
            exact_changed_proposals / changed_proposals if changed_proposals else 0.0
        ),
        "rows_improved_by_character_distance": improved,
        "rows_worsened_by_character_distance": worsened,
        "rows_tied_by_character_distance": tied,
        "original_to_verified_wer": original_word_errors / original_word_units,
        "model_to_verified_wer": model_word_errors / model_word_units,
        "original_to_verified_cer": original_char_errors / original_char_units,
        "model_to_verified_cer": model_char_errors / model_char_units,
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = json.loads(args.truth.expanduser().resolve().read_text(encoding="utf-8"))
    payload = json.loads(
        args.predictions.expanduser().resolve().read_text(encoding="utf-8")
    )
    summary = score_benchmark(truth, payload["rows"])
    summary["model"] = payload.get("model")
    summary["usage"] = payload.get("usage", {})
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
