"""Transparent ASR edit metrics, subgroup views, and bootstrap intervals."""

from __future__ import annotations

import random
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable, Sequence

from sinhala_asr.text.normalizer import metric_normalize


def strict_normalize(text: str) -> str:
    """Apply only Unicode NFC and whitespace cleanup for strict scoring."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int]:
    """Return Levenshtein substitutions, deletions, and insertions."""
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for row_index, reference_item in enumerate(reference, start=1):
        current = [(row_index, 0, row_index, 0)]
        for column_index, hypothesis_item in enumerate(hypothesis, start=1):
            if reference_item == hypothesis_item:
                current.append(previous[column_index - 1])
                continue
            substitution = previous[column_index - 1]
            deletion = previous[column_index]
            insertion = current[column_index - 1]
            choices = [
                (
                    substitution[0] + 1,
                    substitution[1] + 1,
                    substitution[2],
                    substitution[3],
                ),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            current.append(min(choices))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {
        "errors": distance,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_units": len(reference),
    }


def _characters(text: str) -> list[str]:
    return [character for character in text if not character.isspace()]


def score_pair(
    reference: str, prediction: str, normalizer: Callable[[str], str]
) -> dict[str, int]:
    normalized_reference = normalizer(reference)
    normalized_prediction = normalizer(prediction)
    word = edit_counts(normalized_reference.split(), normalized_prediction.split())
    character = edit_counts(
        _characters(normalized_reference), _characters(normalized_prediction)
    )
    return {f"word_{key}": value for key, value in word.items()} | {
        f"character_{key}": value for key, value in character.items()
    }


def aggregate(
    scored_rows: Sequence[dict[str, Any]], prefix: str
) -> dict[str, float | int]:
    totals = Counter()
    for row in scored_rows:
        for unit in ("word", "character"):
            for field in (
                "errors",
                "substitutions",
                "deletions",
                "insertions",
                "reference_units",
            ):
                totals[f"{unit}_{field}"] += int(row[f"{prefix}_{unit}_{field}"])
    words = totals["word_reference_units"]
    characters = totals["character_reference_units"]
    return {
        "rows": len(scored_rows),
        "wer": totals["word_errors"] / words if words else 0.0,
        "cer": totals["character_errors"] / characters if characters else 0.0,
        **dict(totals),
    }


def bootstrap_interval(
    rows: Sequence[dict[str, Any]],
    prefix: str,
    *,
    iterations: int = 1000,
    seed: int = 20260903,
) -> dict[str, list[float]]:
    if not rows or iterations <= 0:
        return {"wer": [0.0, 0.0], "cer": [0.0, 0.0]}
    rng = random.Random(seed)
    values: dict[str, list[float]] = {"wer": [], "cer": []}
    for _ in range(iterations):
        sampled = [rows[rng.randrange(len(rows))] for _ in rows]
        metrics = aggregate(sampled, prefix)
        values["wer"].append(float(metrics["wer"]))
        values["cer"].append(float(metrics["cer"]))
    result = {}
    for metric, samples in values.items():
        samples.sort()
        result[metric] = [
            samples[int(0.025 * (len(samples) - 1))],
            samples[int(0.975 * (len(samples) - 1))],
        ]
    return result


def paired_delta_interval(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    prefix: str,
    *,
    iterations: int = 1000,
    seed: int = 20260903,
) -> dict[str, list[float]]:
    """Bootstrap candidate-minus-baseline WER/CER on aligned rows."""
    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have the same length")
    if not baseline:
        return {"wer": [0.0, 0.0], "cer": [0.0, 0.0]}
    rng = random.Random(seed)
    values: dict[str, list[float]] = {"wer": [], "cer": []}
    for _ in range(iterations):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        baseline_metrics = aggregate([baseline[index] for index in indices], prefix)
        candidate_metrics = aggregate([candidate[index] for index in indices], prefix)
        for metric in values:
            values[metric].append(
                float(candidate_metrics[metric]) - float(baseline_metrics[metric])
            )
    result = {}
    for metric, samples in values.items():
        samples.sort()
        result[metric] = [
            samples[int(0.025 * (len(samples) - 1))],
            samples[int(0.975 * (len(samples) - 1))],
        ]
    return result


def error_labels(
    reference: str, prediction: str, language_class: str | None = None
) -> list[str]:
    strict_reference = strict_normalize(reference)
    strict_prediction = strict_normalize(prediction)
    if strict_reference == strict_prediction:
        return ["exact"]
    labels = []
    if metric_normalize(reference) == metric_normalize(prediction):
        labels.append("normalization_only")
    if "".join(strict_reference.split()) == "".join(strict_prediction.split()):
        labels.append("whitespace_or_segmentation")
    if any(character.isdigit() for character in reference + prediction):
        labels.append("number")
    if language_class == "code_switched":
        labels.append("code_switch")
    counts = edit_counts(strict_reference.split(), strict_prediction.split())
    if counts["substitutions"]:
        labels.append("substitution")
    if counts["deletions"]:
        labels.append("deletion")
    if counts["insertions"]:
        labels.append("insertion")
    if not strict_prediction:
        labels.append("empty_output")
    if len(strict_prediction.split()) > max(8, 2 * len(strict_reference.split())):
        labels.append("possible_repetition_or_hallucination")
    return labels or ["other"]


def evaluate_rows(
    rows: list[dict[str, Any]], *, bootstrap_iterations: int = 1000
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = []
    for row in rows:
        reference = str(row.get("reference") or "")
        prediction = str(row.get("prediction") or "")
        strict = score_pair(reference, prediction, strict_normalize)
        canonical = score_pair(reference, prediction, metric_normalize)
        scored.append(
            dict(row)
            | {f"strict_{key}": value for key, value in strict.items()}
            | {f"canonical_{key}": value for key, value in canonical.items()}
            | {
                "error_labels": error_labels(
                    reference, prediction, row.get("language_class")
                )
            }
        )
    summary: dict[str, Any] = {
        "strict": aggregate(scored, "strict"),
        "canonical": aggregate(scored, "canonical"),
        "confidence_95": {
            "strict": bootstrap_interval(
                scored, "strict", iterations=bootstrap_iterations
            ),
            "canonical": bootstrap_interval(
                scored, "canonical", iterations=bootstrap_iterations
            ),
        },
        "error_label_counts": dict(
            sorted(
                Counter(
                    label for row in scored for label in row["error_labels"]
                ).items()
            )
        ),
    }
    for field in ("source_dataset", "language_class", "dataset_split"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored:
            if row.get(field) is not None:
                groups[str(row[field])].append(row)
        if groups:
            summary[f"by_{field}"] = {
                key: {
                    "strict": aggregate(group, "strict"),
                    "canonical": aggregate(group, "canonical"),
                }
                for key, group in sorted(groups.items())
            }
    return scored, summary
