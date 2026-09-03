#!/usr/bin/env python3
"""Resumably download and safely extract official OpenSLR-52 shards."""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URL = "https://openslr.trmal.net/resources/52"
ALL_SHARDS = tuple("0123456789abcdef")


def download(url: str, final_path: Path) -> Path:
    partial = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.is_file():
        return final_path
    subprocess.run(
        [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "5", "--retry-all-errors",
            "--continue-at", "-", "--output", str(partial), url,
        ],
        check=True,
    )
    partial.replace(final_path)
    return final_path


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe archive member: {member.filename}")
        handle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/openslr52"))
    parser.add_argument("--shards", nargs="+", choices=ALL_SHARDS, default=ALL_SHARDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    small_files = ["LICENSE", "utt_spk_text.tsv"]
    for filename in small_files:
        download(f"{BASE_URL}/{filename}", output / filename)

    def fetch(shard: str) -> Path:
        filename = f"asr_sinhala_{shard}.zip"
        print(f"Downloading shard {shard}", flush=True)
        return download(f"{BASE_URL}/{filename}", output / filename)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        archives = list(pool.map(fetch, args.shards))

    for shard, archive in zip(args.shards, archives, strict=True):
        print(f"Testing and extracting shard {shard}", flush=True)
        with zipfile.ZipFile(archive) as handle:
            bad_member = handle.testzip()
        if bad_member:
            raise ValueError(f"corrupt member {bad_member} in {archive}")
        safe_extract(archive, output)
        (output / f".extracted-{shard}").touch()
        if not args.keep_archives:
            archive.unlink()
    print(f"OpenSLR-52 snapshot ready at {output}")


if __name__ == "__main__":
    main()
