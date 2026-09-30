"""Scan for corrupted/garbage reference transcripts across the WHOLE
stratified_v5 dataset (train + validation + test) -- the kind of thing found
by accident earlier (a literal "cselk" token sitting inside an otherwise
Sinhala reference sentence).

Flags a row when its text contains:
  - Latin letters (a-z/A-Z) mixed into the Sinhala sentence -- legitimate
    exceptions exist (real acronyms, brand names) so this is a candidate
    list for review, not an automatic delete list.
  - Stray unexpected symbols outside normal Sinhala text + punctuation.
  - Suspiciously low Sinhala-character ratio for the row's length (mostly
    non-Sinhala content, e.g. mostly digits/symbols).

Usage:
    python scan_reference_quality.py
"""

import csv
import re

from datasets import load_dataset

DATASET = "Yohan2003/whisper-sl-data"
SPLITS = {
    "train": "data/stratified_v5/train.parquet",
    "validation": "data/stratified_v5/validation.parquet",
    "test": "data/stratified_v5/test.parquet",
}
TEXT_COLUMN = "text"
OUT_PATH = "reference_quality_flags.csv"

_LATIN_RE = re.compile(r"[a-zA-Z]")
_SINHALA_RE = re.compile(r"[඀-෿]")
_ALLOWED_PUNCT_RE = re.compile(r"[\s.,!?;:\"'()\[\]{}।॥඀-෿0-9‌‍]")
MIN_SINHALA_RATIO = 0.5  # below this, the row is mostly not Sinhala text


def classify(text: str) -> list[str]:
    flags = []
    if _LATIN_RE.search(text):
        flags.append("latin_letters")

    non_allowed = _ALLOWED_PUNCT_RE.sub("", text)
    if non_allowed:
        flags.append(f"stray_symbols:{non_allowed[:10]!r}")

    sinhala_chars = len(_SINHALA_RE.findall(text))
    total_chars = len(text.replace(" ", ""))
    if total_chars > 0:
        ratio = sinhala_chars / total_chars
        if ratio < MIN_SINHALA_RATIO:
            flags.append(f"low_sinhala_ratio:{ratio:.2f}")

    return flags


def main():
    rows_out = []
    for split_name, path in SPLITS.items():
        ds = load_dataset(
            "parquet",
            data_files={split_name: f"hf://datasets/{DATASET}/{path}"},
            split=split_name,
            streaming=True,
            columns=[TEXT_COLUMN],
        )
        n = 0
        flagged = 0
        for row in ds:
            n += 1
            text = row[TEXT_COLUMN]
            flags = classify(text)
            if flags:
                flagged += 1
                rows_out.append([split_name, text, ";".join(flags)])
        print(f"{split_name}: {n:,} rows scanned, {flagged:,} flagged")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "text", "flags"])
        writer.writerows(rows_out)

    print(f"\ntotal flagged rows: {len(rows_out):,}")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
