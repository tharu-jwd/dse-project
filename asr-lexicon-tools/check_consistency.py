"""Check whether stratified_v5's transcripts spell the same word two
different ways -- across the WHOLE dataset (train + validation + test), not
just train.

This is the diagnostic that decides your next move:
  - few inconsistencies  -> data is clean, the gap measure_gap.py found is a
                            genuine model weakness -> build the corrector
  - many inconsistencies -> your references contradict each other -> fix the
                            data (majority-vote canonicalisation) and retrain
                            first, corrector second

Prints the top SHOW_TOP conflicts to the terminal as before, and additionally
writes every conflicted group (untruncated) to inconsistencies_full.csv,
sorted by how often the minority spelling appears -- that file is the
priority-ordered list to work through when building the canonical map.

Usage:
    python check_consistency.py
"""

import csv
from collections import Counter, defaultdict

from datasets import load_dataset

from build_soundkey import sound_key, tokenise

DATASET = "SinhaSpeech/sinhala-asr-data"
# Plain parquet under this path, not a registered builder config -- load it
# directly via data_files instead of passing a config name.
SPLITS = {
    "train": "data/stratified_v5/train.parquet",
    "validation": "data/stratified_v5/validation.parquet",
    "test": "data/stratified_v5/test.parquet",
}
TEXT_COLUMN = "text"

SHOW_TOP = 30       # how many worst offenders to print to the terminal
MIN_TOTAL = 5       # ignore groups seen fewer than this many times
OUT_PATH = "inconsistencies_full.csv"


def main():
    groups = defaultdict(Counter)  # sound key -> {spelling: count}

    # streaming=True: skips decoding the audio column entirely (we only touch
    # TEXT_COLUMN below), which is what makes this a text-only, low-bandwidth
    # pull instead of downloading the full audio for each split.
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
                groups[sound_key(word)][word] += 1
        print(f"{split_name}: {n:,} rows scanned")

    total_groups = len(groups)
    conflicted = {
        key: spellings
        for key, spellings in groups.items()
        if len(spellings) > 1 and sum(spellings.values()) >= MIN_TOTAL
    }

    affected_tokens = sum(sum(c.values()) for c in conflicted.values())
    all_tokens = sum(sum(c.values()) for c in groups.values())

    print(f"distinct sound groups      : {total_groups:,}")
    print(f"groups with 2+ spellings   : {len(conflicted):,} "
          f"({len(conflicted) / total_groups:.1%})")
    print(f"word occurrences affected  : {affected_tokens:,} "
          f"({affected_tokens / all_tokens:.1%} of all words)")

    print(f"\ntop {SHOW_TOP} conflicts (by how often the minority form appears):")
    print("-" * 60)

    def minority_count(item):
        counts = item[1].most_common()
        return sum(c for _, c in counts[1:])

    for key, spellings in sorted(conflicted.items(),
                                  key=minority_count, reverse=True)[:SHOW_TOP]:
        parts = "   ".join(f"{w} x{c}" for w, c in spellings.most_common())
        print(f"{key:20s} -> {parts}")

    print("-" * 60)
    print("\nRead the percentages above:")
    print("  under ~1% of words affected -> data is fine, build the corrector")
    print("  above ~3% -> your references contradict each other; fix and retrain")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sound_key", "majority_spelling", "majority_count",
                          "minority_spellings", "minority_total_count", "n_variants"])
        for key, spellings in sorted(conflicted.items(), key=minority_count, reverse=True):
            common = spellings.most_common()
            majority_spelling, majority_count = common[0]
            minority = ";".join(f"{w}:{c}" for w, c in common[1:])
            minority_total = sum(c for _, c in common[1:])
            writer.writerow([key, majority_spelling, majority_count,
                              minority, minority_total, len(common)])

    print(f"\nfull list ({len(conflicted):,} rows, all splits) written to {OUT_PATH}")


if __name__ == "__main__":
    main()
