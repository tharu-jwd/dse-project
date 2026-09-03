"""Strip ZWNJ and suspicious-context ZWJ from the `text` column of both final
datasets, per the findings from `check_hidden_chars.py`:

  - ZWNJ (U+200C) is removed unconditionally -- Sinhala orthography has no
    legitimate use for it (unlike Devanagari/Bengali), so every occurrence
    found was noise.
  - ZWJ (U+200D) is removed only when it's NOT in a valid virama-joined
    conjunct context (immediately preceded by virama U+0DCA and immediately
    followed by a Sinhala consonant, e.g. ශ්‍රේෂ්ඨ). ZWJ that *is* in that
    context is left untouched -- it's load-bearing for correct conjunct
    rendering, not noise.

Classification uses each character's neighbors in the *original* string, so
removing one stray character never changes how another is classified.

Memory handling: `final_collection.parquet` (15k rows) is small enough to
load and clean in one pass. `final_dataset_openslr.parquet` embeds ~15GB of
audio, so it's streamed row-group by row-group via `iter_batches` -- only the
`text` column is ever touched/modified; `audio` bytes pass through each batch
unchanged and are never held beyond that batch.

Usage (from repo root):
    python3 model-development/scripts/clean_hidden_chars.py
"""

import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_COLLECTION_IN = "model-development/data/final_dataset/final_collection.parquet"
DEFAULT_COLLECTION_OUT = "model-development/data/final_dataset/final_collection_cleaned.parquet"
DEFAULT_OPENSLR_IN = "model-development/data/final_dataset/final_dataset_openslr.parquet"
DEFAULT_OPENSLR_OUT = "model-development/data/final_dataset/final_dataset_openslr_cleaned.parquet"

ZWNJ = "‌"
ZWJ = "‍"
VIRAMA = "්"
SINHALA_CONSONANT_RANGE = range(0x0D9A, 0x0DC7)  # U+0D9A-U+0DC6

SCHEMA = pa.schema([
    pa.field("audio", pa.binary()),
    pa.field("source_dataset", pa.string()),
    pa.field("text", pa.string()),
])


class Stats:
    def __init__(self):
        self.rows_seen = 0
        self.rows_changed = 0
        self.zwnj_removed = 0
        self.zwj_removed = 0
        self.zwj_kept = 0

    def report(self, label):
        print(f"\n{label}")
        print(f"  rows: {self.rows_seen}  (changed: {self.rows_changed})")
        print(f"  ZWNJ removed: {self.zwnj_removed}")
        print(f"  ZWJ removed (suspicious context): {self.zwj_removed}")
        print(f"  ZWJ kept (valid conjunct context): {self.zwj_kept}")


def clean_text(text, stats):
    n = len(text)
    out = []
    changed = False
    for i, ch in enumerate(text):
        if ch == ZWNJ:
            stats.zwnj_removed += 1
            changed = True
            continue
        if ch == ZWJ:
            prev_ok = i > 0 and text[i - 1] == VIRAMA
            next_ok = i + 1 < n and ord(text[i + 1]) in SINHALA_CONSONANT_RANGE
            if prev_ok and next_ok:
                stats.zwj_kept += 1
                out.append(ch)
            else:
                stats.zwj_removed += 1
                changed = True
            continue
        out.append(ch)
    stats.rows_seen += 1
    if changed:
        stats.rows_changed += 1
    return "".join(out)


def clean_collection(in_path, out_path):
    df = pd.read_parquet(in_path)
    stats = Stats()
    df["text"] = df["text"].apply(lambda t: clean_text(t, stats))
    df.to_parquet(out_path, index=False)
    stats.report(f"final_collection ({in_path} -> {out_path})")


def clean_openslr(in_path, out_path, batch_size):
    stats = Stats()
    pf = pq.ParquetFile(in_path)
    total = pf.metadata.num_rows
    writer = pq.ParquetWriter(out_path, SCHEMA)
    seen = 0
    try:
        for batch in pf.iter_batches(columns=["audio", "source_dataset", "text"], batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            texts = table.column("text").to_pylist()
            cleaned_texts = [clean_text(t, stats) for t in texts]
            out_table = pa.table(
                {
                    "audio": table.column("audio"),
                    "source_dataset": table.column("source_dataset"),
                    "text": pa.array(cleaned_texts, type=pa.string()),
                },
                schema=SCHEMA,
            )
            writer.write_table(out_table)
            seen += len(texts)
            print(f"  cleaned {seen}/{total} OpenSLR rows", end="\r")
    finally:
        writer.close()
    print()
    stats.report(f"final_dataset_openslr ({in_path} -> {out_path})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection-in", default=DEFAULT_COLLECTION_IN)
    parser.add_argument("--collection-out", default=DEFAULT_COLLECTION_OUT)
    parser.add_argument("--openslr-in", default=DEFAULT_OPENSLR_IN)
    parser.add_argument("--openslr-out", default=DEFAULT_OPENSLR_OUT)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    print(f"Cleaning {args.collection_in} ...")
    clean_collection(args.collection_in, args.collection_out)

    print(f"\nCleaning {args.openslr_in} (streamed) ...")
    clean_openslr(args.openslr_in, args.openslr_out, args.batch_size)


if __name__ == "__main__":
    main()
