"""Split the two final datasets (`final_collection_qa.parquet` = youtube +
bizbrains + linga, `final_dataset_openslr_qa.parquet` = openslr) into two
independent split *configurations*, written to two subfolders:

  stratified/   train.parquet, validation.parquet, test.parquet
      A standard stratified split -- every split gets the same proportion from
      each of the four sources (openslr/youtube/bizbrains/linga) as that source's
      share of the combined pool. Use this for normal training: the model sees
      every domain (including the code-mixed, longer, conversational material
      from `collection`) during training, and test/validation report a blended,
      representative WER.

  held_out/     train.parquet, validation.parquet, held_out_test.parquet
      A cross-domain robustness split. `held_out_test` is carved *only* from the
      collection domain (youtube/bizbrains/linga) -- OpenSLR contributes nothing
      to it -- and is entirely excluded from `train`/`validation` in this
      configuration. The remaining rows (all of OpenSLR + the rest of collection)
      are stratified into train/validation the same way as above. This tells you
      how well a model trained on the bulk of the data generalizes to a domain
      slice it never saw, which a blended stratified test can't measure on its
      own. It's a secondary diagnostic, not the primary split to train your
      shipped model on -- see the `stratified/` configuration for that.

Row order/content is untouched; only membership differs between the two
configurations, and a given row can appear in both (e.g. `stratified/train` and
`held_out/train`) since each configuration is a full, independent partition of
the same underlying pool.

Memory handling: only the lightweight `source_dataset` column is read to decide
every row's split assignments (no audio touched). Both source files are then
streamed once each via `pyarrow.ParquetFile.iter_batches` and every row is
routed to up to two output writers (its stratified-config file and its
held-out-config file); peak memory is bounded by one batch, not either dataset.

Usage (from repo root):
    python3 dse-project/scripts/split_final_datasets.py
    python3 dse-project/scripts/split_final_datasets.py --holdout-frac 0.25 --random-state 0
"""

import argparse
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

DEFAULT_COLLECTION = "dse-project/model-development/data/final_dataset/final_collection_qa.parquet"
DEFAULT_OPENSLR = "dse-project/model-development/data/final_dataset/final_dataset_openslr_qa.parquet"
DEFAULT_OUTPUT_DIR = "dse-project/model-development/data/final_split_dataset"

SCHEMA = pa.schema([
    pa.field("audio", pa.binary()),
    pa.field("source_dataset", pa.string()),
    pa.field("text", pa.string()),
])

STRATIFIED_FILES = {"train": "train.parquet", "validation": "validation.parquet", "test": "test.parquet"}
HELD_OUT_FILES = {"train": "train.parquet", "validation": "validation.parquet", "held_out_test": "held_out_test.parquet"}


def build_row_index(collection_path, openslr_path):
    """Read only `source_dataset` from each file -- cheap, no audio touched."""
    parts = []
    for path in (collection_path, openslr_path):
        col = pd.read_parquet(path, columns=["source_dataset"])
        col["source_file"] = path
        col["row_index"] = range(len(col))
        parts.append(col)
    return pd.concat(parts, ignore_index=True)


def assign_stratified(index_df, train_frac, val_frac, test_frac, random_state):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "stratified fractions must sum to 1.0"
    train_idx, holdout_idx = train_test_split(
        index_df.index, test_size=(val_frac + test_frac),
        stratify=index_df["source_dataset"], random_state=random_state,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=test_frac / (val_frac + test_frac),
        stratify=index_df.loc[holdout_idx, "source_dataset"], random_state=random_state,
    )
    split = pd.Series("train", index=index_df.index)
    split.loc[val_idx] = "validation"
    split.loc[test_idx] = "test"
    return split


def assign_held_out(index_df, openslr_path, holdout_frac, train_frac, val_frac, random_state):
    assert abs(train_frac + val_frac - 1.0) < 1e-9, "held-out train/val fractions must sum to 1.0"
    is_collection = index_df["source_file"] != openslr_path

    collection_idx = index_df.index[is_collection]
    _, held_out_idx = train_test_split(
        collection_idx, test_size=holdout_frac,
        stratify=index_df.loc[collection_idx, "source_dataset"], random_state=random_state,
    )

    split = pd.Series("train", index=index_df.index)
    split.loc[held_out_idx] = "held_out_test"

    pool_idx = index_df.index[split != "held_out_test"]
    train_idx, val_idx = train_test_split(
        pool_idx, test_size=val_frac,
        stratify=index_df.loc[pool_idx, "source_dataset"], random_state=random_state,
    )
    split.loc[train_idx] = "train"
    split.loc[val_idx] = "validation"
    return split


def write_configuration(index_df, split_col, source_files, out_dir, filenames, batch_size):
    os.makedirs(out_dir, exist_ok=True)
    writers = {name: pq.ParquetWriter(os.path.join(out_dir, fname), SCHEMA) for name, fname in filenames.items()}
    counts = {name: 0 for name in filenames}
    try:
        for source_file in source_files:
            assignments = (
                index_df[index_df["source_file"] == source_file]
                .set_index("row_index")[split_col]
            )
            pf = pq.ParquetFile(source_file)
            row_pos = 0
            for batch in pf.iter_batches(columns=["audio", "source_dataset", "text"], batch_size=batch_size):
                n = batch.num_rows
                batch_df = batch.to_pandas()
                batch_df["split"] = assignments.loc[row_pos: row_pos + n - 1].values
                for split_name in filenames:
                    sub = batch_df[batch_df["split"] == split_name]
                    if len(sub) == 0:
                        continue
                    table = pa.table(
                        {
                            "audio": pa.array(sub["audio"], type=pa.binary()),
                            "source_dataset": pa.array(sub["source_dataset"], type=pa.string()),
                            "text": pa.array(sub["text"], type=pa.string()),
                        },
                        schema=SCHEMA,
                    )
                    writers[split_name].write_table(table)
                    counts[split_name] += len(sub)
                row_pos += n
            print(f"  {os.path.basename(source_file)}: streamed {row_pos} rows")
    finally:
        for w in writers.values():
            w.close()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--openslr", default=DEFAULT_OPENSLR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-frac", type=float, default=0.8, help="stratified config")
    parser.add_argument("--val-frac", type=float, default=0.1, help="stratified config")
    parser.add_argument("--test-frac", type=float, default=0.1, help="stratified config")
    parser.add_argument("--holdout-frac", type=float, default=0.2,
                         help="fraction of collection-domain rows reserved as held_out_test")
    parser.add_argument("--heldout-train-frac", type=float, default=0.9, help="of the non-held-out pool")
    parser.add_argument("--heldout-val-frac", type=float, default=0.1, help="of the non-held-out pool")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    print("Building row index (source_dataset only, no audio)...")
    index_df = build_row_index(args.collection, args.openslr)
    print(f"Combined index: {len(index_df)} rows")
    print(index_df["source_dataset"].value_counts())

    print("\nAssigning stratified split (train/validation/test)...")
    index_df["stratified_split"] = assign_stratified(
        index_df, args.train_frac, args.val_frac, args.test_frac, args.random_state
    )
    print(pd.crosstab(index_df["source_dataset"], index_df["stratified_split"]))

    print("\nAssigning held-out split (train/validation/held_out_test)...")
    index_df["heldout_split"] = assign_held_out(
        index_df, args.openslr, args.holdout_frac, args.heldout_train_frac, args.heldout_val_frac, args.random_state
    )
    print(pd.crosstab(index_df["source_dataset"], index_df["heldout_split"]))

    source_files = [args.collection, args.openslr]

    print(f"\nStreaming into {args.output_dir}/stratified/ ...")
    strat_counts = write_configuration(
        index_df, "stratified_split", source_files,
        os.path.join(args.output_dir, "stratified"), STRATIFIED_FILES, args.batch_size,
    )

    print(f"\nStreaming into {args.output_dir}/held_out/ ...")
    heldout_counts = write_configuration(
        index_df, "heldout_split", source_files,
        os.path.join(args.output_dir, "held_out"), HELD_OUT_FILES, args.batch_size,
    )

    print(f"\nDone.")
    print(f"stratified/: {strat_counts}")
    print(f"held_out/:   {heldout_counts}")


if __name__ == "__main__":
    main()
