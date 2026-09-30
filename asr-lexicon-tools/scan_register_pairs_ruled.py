"""Rule-based scan for word-level register pairs (colloquial vs literary
FORMS) across the WHOLE stratified_v5 dataset (train + validation + test).

Replaces the earlier blind edit-distance version (scan_register_candidates.py
-> register_pair_candidates.csv), which produced 131,151 candidates dominated
by false positives -- different grammatical inflections of the same verb
(කරන/කරන්න/කරලා -- "doing"/"do!"/"having done", not interchangeable) or
unrelated short words that happened to be edit-distance-close.

Instead of blind distance, this targets SPECIFIC suffix/pattern pairs
actually observed in run6's error analysis this session -- each one a known
colloquial<->literary alternation, not a grammatical inflection change. Much
narrower, much higher precision, same "candidates for your review" workflow
as before (not auto-resolved -- the canonical_form column is left blank
except where noted).

Usage:
    python scan_register_pairs_ruled.py
"""

import csv
import re
from collections import Counter, defaultdict

from datasets import load_dataset

from build_soundkey import tokenise

DATASET = "Yohan2003/whisper-sl-data"
SPLITS = {
    "train": "data/stratified_v5/train.parquet",
    "validation": "data/stratified_v5/validation.parquet",
    "test": "data/stratified_v5/test.parquet",
}
TEXT_COLUMN = "text"
OUT_PATH = "register_pairs_ruled.csv"
MIN_TOTAL = 3  # ignore pairs seen fewer than this many times combined

# Each rule: (name, suffix_a, suffix_b) -- a word ending in suffix_a is
# considered the same stem as one ending in suffix_b. Both directions are
# checked. These are the specific patterns actually seen in run6's leftover
# errors this session, not a generic rule.
SUFFIX_RULES = [
    ("locative -ේහි/-ේ (archaic/modern locative)", "ේහි", "ේ"),
    ("verb -ණවා/-නවා (present tense epenthetic ැ)", "රනවා", "රේනවා"),
    ("copula -වේ/-වැයි (full/contracted 'is')", "වේ", "වැයි"),
    ("negation -නේ/-අැද්ද (contraction)", "නැහැ", "නැ"),
    ("plural -වරු/-වෝ (formal/colloquial plural)", "වරු", "වෝ"),
    ("conditional -නං/-නම් (nasalized/full ending)", "නං", "නම්"),
]


def stem_variants(word):
    """For a word matching a known suffix, return (rule_name, stem) so
    different-suffix same-stem words can be grouped."""
    for name, sfx_a, sfx_b in SUFFIX_RULES:
        if word.endswith(sfx_a):
            yield name, word[: -len(sfx_a)], "a"
        if word.endswith(sfx_b):
            yield name, word[: -len(sfx_b)], "b"


def main():
    freq = Counter()
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
                freq[word] += 1
        print(f"{split_name}: {n:,} rows scanned")

    print(f"\nvocabulary size: {len(freq):,}")

    # rule_name -> stem -> {word: count}
    groups = defaultdict(lambda: defaultdict(Counter))
    for word, count in freq.items():
        for rule_name, stem, side in stem_variants(word):
            groups[rule_name][stem][word] += count

    rows_out = []
    for rule_name, stems in groups.items():
        for stem, spellings in stems.items():
            if len(spellings) < 2:
                continue
            total = sum(spellings.values())
            if total < MIN_TOTAL:
                continue
            common = spellings.most_common()
            rows_out.append((rule_name, stem, common, total))

    rows_out.sort(key=lambda t: t[3], reverse=True)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "stem", "spellings_seen", "total_count",
                          "same_word_different_register", "canonical_form"])
        for rule_name, stem, common, total in rows_out:
            seen = ";".join(f"{w}:{c}" for w, c in common)
            writer.writerow([rule_name, stem, seen, total, "", ""])

    print(f"\ncandidate register-pair groups found: {len(rows_out):,}")
    print(f"written to {OUT_PATH}")
    print()
    by_rule = Counter(r[0] for r in rows_out)
    for rule_name, count in by_rule.most_common():
        print(f"  {rule_name}: {count} groups")


if __name__ == "__main__":
    main()
