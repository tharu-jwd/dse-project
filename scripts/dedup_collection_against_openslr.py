"""Cross-check `final_collection.parquet` (Linga + YouTube + BizBrains) against
`final_dataset_openslr.parquet` and drop any collection row whose audio is a
byte-for-byte duplicate of a row already present in OpenSLR-52.

Why audio bytes, not text: the overlap found between `linga` and OpenSLR isn't just
matching transcripts -- sampled pairs had identical MD5 hashes on the raw WAV bytes,
i.e. the exact same recording is present in both sources under different
`source_dataset` labels. Matching on text alone would also flag distinct recordings
of the same short prompt sentence read by different speakers (common in OpenSLR,
not a real duplicate), so this hashes audio content instead -- no false positives
from shared prompts, no false negatives from transcript-cleaning differences.

Memory handling: `final_dataset_openslr.parquet` embeds ~15GB of WAV bytes, too big
to load into one DataFrame. This script streams it row-group by row-group via
`pyarrow.ParquetFile.iter_batches`, hashing each row's audio and keeping only the
32-char MD5 hex digest in memory (150k rows x ~32 bytes, a few MB total) -- the raw
audio bytes are never held past the batch they arrived in.
`final_collection.parquet` is small enough (15k rows, ~2GB) to load whole.

Usage (from repo root):
    python3 dse-project/scripts/dedup_collection_against_openslr.py
    python3 dse-project/scripts/dedup_collection_against_openslr.py \
        --collection dse-project/model-development/data/final_dataset/final_collection.parquet \
        --openslr dse-project/model-development/data/final_dataset/final_dataset_openslr.parquet \
        --output dse-project/model-development/data/final_dataset/final_collection_deduped.parquet
"""

import argparse
import hashlib

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_COLLECTION = "dse-project/model-development/data/final_dataset/final_collection.parquet"
DEFAULT_OPENSLR = "dse-project/model-development/data/final_dataset/final_dataset_openslr.parquet"
DEFAULT_OUTPUT = "dse-project/model-development/data/final_dataset/final_collection_deduped.parquet"


def hash_openslr_audio(path, batch_size):
    """Stream OpenSLR's audio column; return the set of MD5 hex digests seen."""
    hashes = set()
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    seen = 0
    for batch in pf.iter_batches(columns=["audio"], batch_size=batch_size):
        for audio_bytes in batch.column("audio"):
            hashes.add(hashlib.md5(audio_bytes.as_py()).hexdigest())
        seen += batch.num_rows
        print(f"  hashed {seen}/{total} OpenSLR rows", end="\r")
    print()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--openslr", default=DEFAULT_OPENSLR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2000, help="rows per streamed batch when hashing OpenSLR")
    args = parser.parse_args()

    print(f"Streaming + hashing OpenSLR audio from {args.openslr} ...")
    openslr_hashes = hash_openslr_audio(args.openslr, args.batch_size)
    print(f"OpenSLR unique audio hashes: {len(openslr_hashes)}")

    print(f"\nLoading collection dataset from {args.collection} ...")
    df = pd.read_parquet(args.collection)
    print(f"Collection rows: {len(df)}")
    print(df["source_dataset"].value_counts())

    print("\nHashing collection audio and cross-checking against OpenSLR ...")
    df["audio_md5"] = df["audio"].apply(lambda b: hashlib.md5(b).hexdigest())
    is_dup = df["audio_md5"].isin(openslr_hashes)

    print(f"\nDuplicate rows found (also present in OpenSLR): {is_dup.sum()} / {len(df)}")
    print(df.loc[is_dup, "source_dataset"].value_counts())

    # Informational only: duplicates *within* the collection itself, not touched here.
    internal_dup = df["audio_md5"].duplicated(keep=False) & ~is_dup
    if internal_dup.any():
        print(f"\n[note] {internal_dup.sum()} rows are also duplicated *within* the "
              f"collection itself (left untouched by this script):")
        print(df.loc[internal_dup, "source_dataset"].value_counts())

    deduped = df.loc[~is_dup].drop(columns=["audio_md5"]).reset_index(drop=True)
    print(f"\nRemaining rows after dropping OpenSLR duplicates: {len(deduped)}")
    print(deduped["source_dataset"].value_counts())

    deduped.to_parquet(args.output, index=False)
    print(f"\nSaved deduped collection to {args.output}")


if __name__ == "__main__":
    main()
