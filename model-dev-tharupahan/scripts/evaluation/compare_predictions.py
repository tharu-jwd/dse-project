#!/usr/bin/env python3
"""Compare two aligned ASR prediction files with paired uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sinhala_asr.evaluation.metrics import (
    aggregate,
    evaluate_rows,
    paired_delta_interval,
)

from evaluate_predictions import read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    baseline_by_id = {row["sample_id"]: row for row in read_rows(args.baseline)}
    candidate_by_id = {row["sample_id"]: row for row in read_rows(args.candidate)}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise SystemExit("baseline and candidate sample IDs differ")
    ids = sorted(baseline_by_id)
    if any(
        baseline_by_id[key]["reference"] != candidate_by_id[key]["reference"]
        for key in ids
    ):
        raise SystemExit("baseline and candidate references differ")

    baseline, _ = evaluate_rows(
        [baseline_by_id[key] for key in ids], bootstrap_iterations=0
    )
    candidate, _ = evaluate_rows(
        [candidate_by_id[key] for key in ids], bootstrap_iterations=0
    )
    summary: dict[str, object] = {"rows": len(ids)}
    for prefix in ("strict", "canonical"):
        old = aggregate(baseline, prefix)
        new = aggregate(candidate, prefix)
        summary[prefix] = {
            "baseline": old,
            "candidate": new,
            "candidate_minus_baseline": {
                "wer": float(new["wer"]) - float(old["wer"]),
                "cer": float(new["cer"]) - float(old["cer"]),
            },
            "candidate_minus_baseline_confidence_95": paired_delta_interval(
                baseline,
                candidate,
                prefix,
                iterations=args.bootstrap_iterations,
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
