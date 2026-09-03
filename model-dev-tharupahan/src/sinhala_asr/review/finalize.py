"""Apply native adjudications and lock an immutable dataset version."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from sinhala_asr.text.normalizer import canonicalize, metric_normalize


def finalize_rows(
    rows: list[dict[str, Any]], adjudications: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_ids = {
        str(row["sample_id"])
        for row in rows
        if row.get("dataset_split") in {"validation_candidate", "test_candidate"}
    }
    missing = sorted(candidate_ids - set(adjudications))
    if missing:
        raise ValueError(
            f"{len(missing)} gold candidates are not adjudicated; first: {missing[0]}"
        )
    unknown = sorted(set(adjudications) - {str(row["sample_id"]) for row in rows})
    if unknown:
        raise ValueError(
            f"adjudications contain unknown sample IDs; first: {unknown[0]}"
        )

    finalized = []
    for row in rows:
        item = dict(row)
        split = item.get("dataset_split")
        if split not in {"validation_candidate", "test_candidate"}:
            finalized.append(item)
            continue
        record = adjudications[str(item["sample_id"])]
        decision = str(record["decision"])
        item["adjudication_decision"] = decision
        item["adjudication_notes"] = str(record.get("notes") or "")
        item["reviewed_at_utc"] = record.get("reviewed_at_utc")
        if decision in {"correct", "edited"}:
            corrected = str(record.get("text_corrected") or "").strip()
            if not corrected:
                raise ValueError(
                    f"accepted sample {item['sample_id']} has an empty transcript"
                )
            if (
                decision == "edited"
                and corrected == str(item.get("text_original") or "").strip()
            ):
                raise ValueError(
                    f"edited sample {item['sample_id']} has unchanged text"
                )
            item["text_reviewed"] = corrected
            item["text_canonical"] = canonicalize(corrected)
            item["text_metric"] = metric_normalize(corrected)
            item["dataset_split"] = split.removesuffix("_candidate")
            item["exclusion_reason"] = None
        else:
            item["text_reviewed"] = None
            item["dataset_split"] = "excluded"
            item["exclusion_reason"] = f"native_review_{decision}"
        finalized.append(item)

    counts = Counter(str(row["dataset_split"]) for row in finalized)
    fingerprint_input = "\n".join(
        f"{row['sample_id']}\t{row['dataset_split']}\t{row.get('text_canonical', '')}"
        for row in sorted(finalized, key=lambda value: value["sample_id"])
    )
    summary = {
        "split_counts": dict(sorted(counts.items())),
        "adjudicated_candidates": len(candidate_ids),
        "dataset_fingerprint": hashlib.sha256(fingerprint_input.encode()).hexdigest(),
    }
    return finalized, summary
