"""Conservative, versioned Sinhala transcript normalization.

Canonicalization fixes representation and formatting only. It deliberately does
not merge compounds, expand colloquial forms, transliterate English, or replace
spelling variants; those decisions require a reviewed linguistic policy.
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "si-conservative-v1"
SINHALA_VIRAMA = "්"
VALID_ZWJ_FOLLOWERS = frozenset("රයෂ")
_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?%])")


def _is_valid_zwj(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1] == SINHALA_VIRAMA
        and text[index + 1] in VALID_ZWJ_FOLLOWERS
    )


def canonicalize(text: str) -> str:
    """Return a stable training representation without semantic rewriting."""
    if not isinstance(text, str):
        raise TypeError("transcript must be a string")

    text = unicodedata.normalize("NFC", text)
    cleaned: list[str] = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        if char == "\u200d":
            if _is_valid_zwj(text, index):
                cleaned.append(char)
            continue
        if category in {"Cc", "Cf"} and char not in {"\n", "\r", "\t"}:
            continue
        cleaned.append(char)

    normalized = _WHITESPACE.sub(" ", "".join(cleaned)).strip()
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", normalized)


def metric_normalize(text: str) -> str:
    """Normalize formatting for canonical WER/CER while preserving words."""
    canonical = canonicalize(text).lower()
    characters = [
        " " if unicodedata.category(char).startswith("P") else char
        for char in canonical
    ]
    return _WHITESPACE.sub(" ", "".join(characters)).strip()


def transcript_flags(text: object) -> list[str]:
    """Return deterministic quality flags without changing the source text."""
    if not isinstance(text, str):
        return ["non_string_transcript"]

    flags: list[str] = []
    if not text.strip():
        flags.append("empty_transcript")
    if text != unicodedata.normalize("NFC", text):
        flags.append("non_nfc")
    if any(unicodedata.category(char) in {"Cc", "Cf"} for char in text):
        flags.append("control_or_format_character")
    if any(char == "\u200d" and not _is_valid_zwj(text, i) for i, char in enumerate(text)):
        flags.append("invalid_zwj")
    if re.search(r"[A-Za-z]", text):
        flags.append("code_switched_latin")
    if any(char.isdigit() for char in text):
        flags.append("contains_digit")
    return flags

