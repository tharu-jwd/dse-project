#!/usr/bin/env python3
"""
Download and extract the full OpenSLR SLR52 (Sinhala ASR) dataset.

Dataset info: https://openslr.org/52/
- 16 zip shards: asr_sinhala_0.zip ... asr_sinhala_9.zip, asr_sinhala_a.zip ... asr_sinhala_f.zip
- 1 transcript file: utt_spk_text.tsv
- Total size ~15GB

Usage:
    python download_openslr_52.py
    python download_openslr_52.py --output-dir /path/to/save --skip-extract
    python download_openslr_52.py --shards 0 1 2   # download only specific shards
"""

import os
import sys
import argparse
import zipfile
import urllib.request
import urllib.error

BASE_URL = "https://openslr.trmal.net/resources/52/"
ALL_SHARDS = list("012345-6789abcdef")
TRANSCRIPT_FILE = "utt_spk_text.tsv"


def format_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        sys.stdout.write(
            f"\r    {format_size(downloaded)} / {format_size(total_size)} ({pct:.1f}%)"
        )
    else:
        sys.stdout.write(f"\r    {format_size(downloaded)}")
    sys.stdout.flush()


def download_file(filename, output_dir, retries=3):
    url = BASE_URL + filename
    dest = os.path.join(output_dir, filename)

    if os.path.exists(dest):
        print(f"[skip] {filename} already downloaded.")
        return dest

    tmp_dest = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            print(f"[download] {filename} (attempt {attempt}/{retries})")
            urllib.request.urlretrieve(url, tmp_dest, reporthook=show_progress)
            print()  # newline after progress bar
            os.rename(tmp_dest, dest)
            return dest
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f"\n[error] Failed to download {filename}: {e}")
            if attempt == retries:
                print(f"[fail] Giving up on {filename} after {retries} attempts.")
                return None

    return None


def extract_zip(zip_path, output_dir):
    if zip_path is None:
        return
    print(f"[extract] {os.path.basename(zip_path)} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        print(f"[done] Extracted {os.path.basename(zip_path)}")
    except zipfile.BadZipFile:
        print(f"[error] {zip_path} is corrupted. Delete it and re-download.")


def main():
    parser = argparse.ArgumentParser(description="Download OpenSLR SLR52 Sinhala ASR dataset")
    parser.add_argument(
        "--output-dir", default="./openslr_52",
        help="Directory to save/extract files (default: ./openslr_52)"
    )
    parser.add_argument(
        "--shards", nargs="+", default=ALL_SHARDS,
        help="Specific shards to download, e.g. --shards 0 1 a f (default: all 16)"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Only download zip files, skip extraction"
    )
    parser.add_argument(
        "--keep-zips", action="store_true",
        help="Keep zip files after extraction (default: keeps them; add --delete-zips to remove)"
    )
    parser.add_argument(
        "--delete-zips", action="store_true",
        help="Delete zip files after successful extraction to save disk space"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Download transcript file
    print("=== Transcript file ===")
    download_file(TRANSCRIPT_FILE, args.output_dir)

    # 2. Download (and optionally extract) each shard
    print("\n=== Audio shards ===")
    for shard in args.shards:
        filename = f"asr_sinhala_{shard}.zip"
        zip_path = download_file(filename, args.output_dir)

        if zip_path and not args.skip_extract:
            extract_zip(zip_path, args.output_dir)
            if args.delete_zips:
                os.remove(zip_path)
                print(f"[cleanup] Removed {filename}")

    print("\nAll requested files processed.")
    print(f"Data is available in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
