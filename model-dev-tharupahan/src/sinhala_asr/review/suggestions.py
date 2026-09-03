"""Validate and classify external text-only transcript suggestions."""

from __future__ import annotations

from sinhala_asr.text.normalizer import metric_normalize


def classify_change(original: str, suggestion: str) -> str:
    if original == suggestion:
        return "unchanged"
    if metric_normalize(original) == metric_normalize(suggestion):
        return "format_case_punctuation_only"
    compact_original = "".join(metric_normalize(original).split())
    compact_suggestion = "".join(metric_normalize(suggestion).split())
    if compact_original == compact_suggestion:
        return "spacing_only"
    return "lexical_or_spelling"
