"""Systematic scan for ZWJ/conjunct-joiner inconsistency across the WHOLE
stratified_v5 dataset (train + validation + test) -- the proper version of
what was found by accident earlier (20 rows discovered as a side effect of
the letter-spelling scan, because tokenise() strips ZWJ before comparing
words and so happened to surface a few ZWJ-affected pairs as false letter
"matches").

This scans the RAW text (ZWJ kept, not stripped) for two conjunct types that
require a zero-width joiner (U+200D) to render correctly in Sinhala:

  - rakaransaya (subscript ra, "touching r"):  consonant + virama + ZWJ + ra
    e.g. correct: ප්‍ර (ප්ර+ZWJ -> renders as ප්Ȁdර)  wrong (no ZWJ): ප්ර
  - yansaya (subscript ya):                    consonant + virama + ZWJ + ya
    e.g. correct: ව්‍ය          wrong (no ZWJ): ව්ය

For each word, strips ZWJ to get a "conjunct key" so both spellings (with
and without the joiner) group together, then reports how often each group
has 2+ actual spellings -- same diagnostic shape as check_consistency.py,
just targeting the joiner instead of letter substitution.

The grammar rule here is unconditional (unlike the retroflex n/N letter
issue): the ZWJ-joined form is ALWAYS the correct one for these two conjunct
types, so the correction column is auto-filled, not left for manual review.

Usage:
    python check_zwj_consistency.py
"""

import csv
import re
from collections import Counter, defaultdict

from datasets import load_dataset

DATASET = "Yohan2003/whisper-sl-data"
SPLITS = {
    "train": "data/stratified_v5/train.parquet",
    "validation": "data/stratified_v5/validation.parquet",
    "test": "data/stratified_v5/test.parquet",
}
TEXT_COLUMN = "text"
OUT_PATH = "zwj_conjunct_inconsistencies.csv"
SHOW_TOP = 30

ZWJ = "‍"
VIRAMA = "්"
RA = "ර"
YA = "ය"

# Matches a virama immediately followed by ra or ya, with or without a ZWJ
# in between -- this is the pattern that needs the joiner to render as the
# proper rakaransaya/yansaya conjunct instead of a plain separate letter.
_CONJUNCT_RE = re.compile(f"{VIRAMA}{ZWJ}?[{RA}{YA}]")
_STRIP_PUNCT_RE = re.compile(r"[.,!?;:\"'()\[\]{}।॥]")


def conjunct_key(word: str) -> str:
    """Strip ZWJ so both spellings of the same word group together."""
    return word.replace(ZWJ, "")


def tokenise(text: str) -> list[str]:
    text = _STRIP_PUNCT_RE.sub("", text)
    return [w for w in text.split() if w]


def force_joiner(word: str) -> str:
    """Insert ZWJ into every virama+ra/ya sequence that's missing one --
    this is the grammatically-correct form, used as the auto-correction."""
    return _CONJUNCT_RE.sub(lambda m: m.group(0) if ZWJ in m.group(0)
                             else m.group(0)[0] + ZWJ + m.group(0)[1], word)


def main():
    groups = defaultdict(Counter)  # conjunct key -> {actual spelling: count}

    for split_name, path in SPLITS.items():
        ds = load_dataset(
            "parquet",
            data_files={split_name: f"hf://datasets/{DATASET}/{path}"},
            split=split_name,
            streaming=True,
            columns=[TEXT_COLUMN],
        )
        n = 0
        for row in ds:
            n += 1
            for word in tokenise(row[TEXT_COLUMN]):
                if _CONJUNCT_RE.search(word):
                    groups[conjunct_key(word)][word] += 1
        print(f"{split_name}: {n:,} rows scanned")

    total_groups = len(groups)
    conflicted = {k: v for k, v in groups.items() if len(v) > 1}
    affected_tokens = sum(sum(c.values()) for c in conflicted.values())
    all_conjunct_tokens = sum(sum(c.values()) for c in groups.values())

    print(f"\nwords containing a rakaransaya/yansaya conjunct: {total_groups:,}")
    print(f"groups with both ZWJ and non-ZWJ spellings present: {len(conflicted):,} "
          f"({len(conflicted)/total_groups:.1%})")
    print(f"word occurrences affected: {affected_tokens:,} "
          f"({affected_tokens/all_conjunct_tokens:.1%} of all conjunct-containing words)")

    print(f"\ntop {SHOW_TOP} conflicts (by how often the wrong/non-ZWJ form appears):")
    print("-" * 60)

    def wrong_count(item):
        return sum(c for w, c in item[1].items() if ZWJ not in w)

    ranked = sorted(conflicted.items(), key=wrong_count, reverse=True)
    for key, spellings in ranked[:SHOW_TOP]:
        parts = "   ".join(f"{w!r} x{c}" for w, c in spellings.most_common())
        print(f"{key:20s} -> {parts}")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["conjunct_key", "spellings_seen", "total_count",
                          "non_zwj_count", "correction"])
        for key, spellings in ranked:
            seen = ";".join(f"{w}:{c}" for w, c in spellings.most_common())
            total = sum(spellings.values())
            non_zwj = wrong_count((key, spellings))
            writer.writerow([key, seen, total, non_zwj, force_joiner(key)])

    print("-" * 60)
    print(f"\nfull list ({len(conflicted):,} rows, all splits) written to {OUT_PATH}")
    print("correction column is auto-filled (unconditional grammar rule -- the")
    print("ZWJ-joined form is always correct for these two conjunct types).")


if __name__ == "__main__":
    main()
