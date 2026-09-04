#!/usr/bin/env python3
"""Run resumable Bedrock transcript refinement over bounded JSON batches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    rows = json.loads(args.input.expanduser().resolve().read_text(encoding="utf-8"))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("refine_transcripts_bedrock.py")
    for offset in range(0, len(rows), args.batch_size):
        number = offset // args.batch_size
        batch_input = output / f"batch-{number:04d}-input.json"
        batch_result = output / f"batch-{number:04d}-result.json"
        batch_raw = output / f"batch-{number:04d}-raw.txt"
        if batch_result.is_file():
            print(f"Skipping completed {batch_result.name}")
            continue
        batch_input.write_text(
            json.dumps(
                rows[offset : offset + args.batch_size], ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--input",
                str(batch_input),
                "--output",
                str(batch_result),
                "--raw-output",
                str(batch_raw),
                "--model",
                args.model,
                "--region",
                args.region,
            ],
            check=True,
        )
    results = []
    usage: dict[str, int] = {}
    for number in range((len(rows) + args.batch_size - 1) // args.batch_size):
        payload = json.loads((output / f"batch-{number:04d}-result.json").read_text())
        results.extend(payload["rows"])
        for key, value in payload.get("usage", {}).items():
            usage[key] = usage.get(key, 0) + int(value)
    if len(results) != len(rows):
        raise SystemExit(f"expected {len(rows)} results, found {len(results)}")
    combined = {
        "model": args.model,
        "region": args.region,
        "source_rows": len(rows),
        "changed_rows": sum(bool(row.get("changed")) for row in results),
        "usage": usage,
        "rows": results,
    }
    (output / "combined.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: combined[key] for key in ("source_rows", "changed_rows", "usage")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
