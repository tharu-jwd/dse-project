"""Split the combined final dataset (Linga + YouTube + BizBrains + OpenSLR-52) into
train / validation / test parquet files, stratified by `source_dataset`.

A plain random split could, by chance, leave test/validation dominated by whichever
source happens to be largest (OpenSLR-52 is ~91% of the combined row count), or miss
a smaller source entirely. Stratifying on `source_dataset` instead guarantees every
split gets the same *proportion* from each source as that source's share of the
whole dataset -- e.g. if BizBrains is 3% of all rows, it's ~3% of train, ~3% of
validation, and ~3% of test too.

Memory handling: the source parquets embed WAV bytes per row and OpenSLR-52 alone is
~150k rows -- loading everything into one DataFrame just to shuffle and re-save it
would mean holding the entire embedded-audio dataset in memory at once. Instead this
script:
  1. Reads only the lightweight `source_dataset` column from each source file to
     decide the train/validation/test assignment per row (no audio touched).
  2. Streams each source file back through in batches (`pyarrow.ParquetFile.
     iter_batches`), routing each row to the correct split's `ParquetWriter` by its
     precomputed assignment. Peak memory is bounded by one batch, not the whole
     dataset.

Each output file keeps the same three columns as the source files -- `audio`,
`source_dataset`, `text` -- so `source_dataset` stays available in every split to
identify which source each row came from.

Usage (run from the repo root):
    python3 dse-project/scripts/split_dataset.py
    python3 dse-project/scripts/split_dataset.py --train-frac 0.7 --val-frac 0.15 --test-frac 0.15
    python3 dse-project/scripts/split_dataset.py --source-files path/a.parquet path/b.parquet
"""

import argparse
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

DEFAULT_FINAL_DIR = "dse-project/model-development/data/final_dataset"
DEFAULT_OUTPUT_DIR = "dse-project/model-development/data/final_split_dataset"
DEFAULT_SOURCE_FILENAMES = ["final_dataset.parquet", "final_dataset_openslr.parquet"]

SPLIT_NAMES = ["train", "validation", "test"]
SCHEMA = pa.schema([
    pa.field("audio", pa.binary()),
    pa.field("source_dataset", pa.string()),
    pa.field("text", pa.string()),
])


def build_row_index(source_files):
    """Reads only `source_dataset` from each source file -- cheap, no audio touched."""
    index_parts = []
    for path in source_files:
        src_col = pd.read_parquet(path, columns=["source_dataset"])
        src_col["source_file"] = path
        src_col["row_index"] = range(len(src_col))
        index_parts.append(src_col)
    return pd.concat(index_parts, ignore_index=True)


def assign_splits(index_df, train_frac, val_frac, test_frac, random_state):
    """Stratified by source_dataset: carve off test, then split the remainder."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "fractions must sum to 1.0"

    train_idx, holdout_idx = train_test_split(
        index_df.index,
        test_size=(val_frac + test_frac),
        stratify=index_df["source_dataset"],
        random_state=random_state,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=test_frac / (val_frac + test_frac),
        stratify=index_df.loc[holdout_idx, "source_dataset"],
        random_state=random_state,
    )

    index_df = index_df.copy()
    index_df["split"] = "train"
    index_df.loc[val_idx, "split"] = "validation"
    index_df.loc[test_idx, "split"] = "test"
    return index_df


def write_splits(index_df, source_files, output_dir, batch_size):
    """Streams each source file through in batches, routing rows to the right split's writer."""
    os.makedirs(output_dir, exist_ok=True)
    writers = {
        split: pq.ParquetWriter(os.path.join(output_dir, f"{split}.parquet"), SCHEMA)
        for split in SPLIT_NAMES
    }
    split_counts = {split: 0 for split in SPLIT_NAMES}

    try:
        for source_file in source_files:
            file_split = (
                index_df[index_df["source_file"] == source_file]
                .set_index("row_index")["split"]
            )
            pf = pq.ParquetFile(source_file)
            row_pos = 0
            for batch in pf.iter_batches(batch_size=batch_size):
                batch_len = batch.num_rows
                batch_df = batch.to_pandas()
                batch_df["split"] = file_split.loc[row_pos: row_pos + batch_len - 1].values

                for split in SPLIT_NAMES:
                    sub = batch_df[batch_df["split"] == split]
                    if len(sub) == 0:
                        continue
                    sub_table = pa.table(
                        {
                            "audio": pa.array(sub["audio"], type=pa.binary()),
                            "source_dataset": pa.array(sub["source_dataset"], type=pa.string()),
                            "text": pa.array(sub["text"], type=pa.string()),
                        },
                        schema=SCHEMA,
                    )
                    writers[split].write_table(sub_table)
                    split_counts[split] += len(sub)

                row_pos += batch_len
            print(f"{os.path.basename(source_file)}: streamed {row_pos} rows")
    finally:
        for w in writers.values():
            w.close()

    return split_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final-dir", default=DEFAULT_FINAL_DIR, help="folder containing the source parquet files")
    parser.add_argument(
        "--source-files",
        nargs="+",
        default=None,
        help=f"explicit source parquet paths (default: {DEFAULT_SOURCE_FILENAMES} inside --final-dir)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2000, help="rows per read/write batch")
    args = parser.parse_args()

    source_files = args.source_files or [os.path.join(args.final_dir, name) for name in DEFAULT_SOURCE_FILENAMES]
    missing = [p for p in source_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Source file(s) not found: {missing}")

    print("Building row index (source_dataset only, no audio)...")
    index_df = build_row_index(source_files)
    print(f"Combined index: {len(index_df)} rows across {len(source_files)} source file(s)")
    print(index_df["source_dataset"].value_counts())

    print("\nAssigning splits (stratified by source_dataset)...")
    index_df = assign_splits(index_df, args.train_frac, args.val_frac, args.test_frac, args.random_state)
    print(pd.crosstab(index_df["source_dataset"], index_df["split"]))
    print("\nPer-source proportions across splits (%, should match within each column):")
    print((pd.crosstab(index_df["source_dataset"], index_df["split"], normalize="columns") * 100).round(2))

    print(f"\nStreaming rows into {args.output_dir}/ (batch size {args.batch_size})...")
    split_counts = write_splits(index_df, source_files, args.output_dir, args.batch_size)

    print(f"\nDone. Final split row counts: {split_counts}")
    for split in SPLIT_NAMES:
        path = os.path.join(args.output_dir, f"{split}.parquet")
        print(f"  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
