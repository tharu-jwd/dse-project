"""Scan the `text` column of the final datasets for invisible/formatting Unicode
characters that have no business being in a clean ASR transcript: zero-width
joiners/spaces, NBSP, BOM, soft hyphen, directional marks, and any other
Unicode "Cf" (format) or "Cc" (control) codepoint picked up via `unicodedata`
rather than a fixed list, so nothing unexpected slips past.

ZWJ (U+200D) gets special treatment: it's not inherently wrong in Sinhala text
-- it's the character that makes virama-joined conjuncts render correctly
(consonant + virama + ZWJ + consonant, e.g. ක්‍ර). So each ZWJ occurrence is
classified as "plausible" (immediately preceded by virama U+0DCA and followed
by a Sinhala consonant) or "suspicious" (anywhere else -- start/end of string,
next to whitespace, doubled, etc.) instead of being flagged outright.

Read-only: only loads the `text`/`source_dataset` columns (audio is never
touched), so this is cheap and safe to run against both final datasets.

Usage (from repo root):
    python3 dse-project/scripts/check_hidden_chars.py
    python3 dse-project/scripts/check_hidden_chars.py --examples 5
"""

import argparse
import unicodedata
from collections import Counter

import pandas as pd

DEFAULT_FILES = {
    "final_collection": "dse-project/model-development/data/final_dataset/final_collection.parquet",
    "final_dataset_openslr": "dse-project/model-development/data/final_dataset/final_dataset_openslr.parquet",
}

# Explicit named list (covers the ones asked about, plus close relatives) --
# reported by name even though the generic Cf/Cc sweep below would also catch them.
NAMED_CHARS = {
    "​": "ZERO WIDTH SPACE (ZWSP)",
    "‌": "ZERO WIDTH NON-JOINER (ZWNJ)",
    "‍": "ZERO WIDTH JOINER (ZWJ)",
    " ": "NO-BREAK SPACE (NBSP)",
    "﻿": "ZERO WIDTH NO-BREAK SPACE / BOM",
    "­": "SOFT HYPHEN",
    "‎": "LEFT-TO-RIGHT MARK (LRM)",
    "‏": "RIGHT-TO-LEFT MARK (RLM)",
    "⁠": "WORD JOINER",
    "�": "REPLACEMENT CHARACTER",
    " ": "LINE SEPARATOR",
    " ": "PARAGRAPH SEPARATOR",
}

SINHALA_CONSONANT = range(0x0D9A, 0x0DC7)  # U+0D9A-U+0DC6
VIRAMA = "්"


def classify_zwj(text, idx):
    prev_ok = idx > 0 and text[idx - 1] == VIRAMA
    next_ok = idx + 1 < len(text) and ord(text[idx + 1]) in SINHALA_CONSONANT
    return "plausible" if (prev_ok and next_ok) else "suspicious"


def context(text, idx, radius=8):
    lo, hi = max(0, idx - radius), min(len(text), idx + radius + 1)
    return text[lo:idx] + "⟦" + text[idx] + "⟧" + text[idx + 1:hi]


def scan(df, examples_per_char):
    named_counts = Counter()
    named_rows = Counter()
    zwj_class_counts = Counter()
    other_cf_cc = Counter()
    other_rows = Counter()
    examples = {}

    for text in df["text"]:
        seen_named_this_row = set()
        for i, ch in enumerate(text):
            if ch in NAMED_CHARS:
                named_counts[ch] += 1
                seen_named_this_row.add(ch)
                if ch == "‍":
                    zwj_class_counts[classify_zwj(text, i)] += 1
                key = ch
                examples.setdefault(key, [])
                if len(examples[key]) < examples_per_char:
                    examples[key].append(context(text, i))
                continue

            cat = unicodedata.category(ch)
            if cat in ("Cf", "Cc", "Co", "Cs") and ch not in NAMED_CHARS:
                other_cf_cc[(ch, cat, unicodedata.name(ch, "UNNAMED"))] += 1
                other_rows[ch] += 1
                examples.setdefault(ch, [])
                if len(examples[ch]) < examples_per_char:
                    examples[ch].append(context(text, i))

        for ch in seen_named_this_row:
            named_rows[ch] += 1

    return named_counts, named_rows, zwj_class_counts, other_cf_cc, other_rows, examples


def report(label, df, examples_per_char):
    print(f"\n{'=' * 70}\n{label}  ({len(df)} rows)\n{'=' * 70}")
    named_counts, named_rows, zwj_class, other, other_rows, examples = scan(df, examples_per_char)

    if not named_counts and not other:
        print("Clean -- no hidden/format/control characters found.")
        return

    if named_counts:
        print(f"\n{'char':<8}{'name':<38}{'occurrences':<14}{'rows affected'}")
        for ch, name in NAMED_CHARS.items():
            if named_counts[ch]:
                print(f"U+{ord(ch):04X}  {name:<36}{named_counts[ch]:<14}{named_rows[ch]}")
                for ex in examples.get(ch, []):
                    print(f"    e.g. ...{ex}...")

        if named_counts["‍"]:
            print(f"\n  ZWJ breakdown: {zwj_class['plausible']} plausible (valid conjunct context), "
                  f"{zwj_class['suspicious']} suspicious (not a virama-joined conjunct)")

    if other:
        print(f"\nOther Unicode format/control/private-use/surrogate characters found:")
        print(f"{'char':<10}{'category':<10}{'name':<40}{'occurrences':<14}{'rows'}")
        for (ch, cat, name), count in sorted(other.items(), key=lambda kv: -kv[1]):
            print(f"U+{ord(ch):04X}    {cat:<10}{name:<40}{count:<14}{other_rows[ch]}")
            for ex in examples.get(ch, []):
                print(f"    e.g. ...{ex}...")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", type=int, default=3, help="example snippets to print per character")
    parser.add_argument("--files", nargs="+", default=None,
                         help="explicit label=path pairs; default scans both final datasets")
    args = parser.parse_args()

    files = DEFAULT_FILES
    if args.files:
        files = dict(f.split("=", 1) for f in args.files)

    for label, path in files.items():
        df = pd.read_parquet(path, columns=["text"])
        report(label, df, args.examples)


if __name__ == "__main__":
    main()
