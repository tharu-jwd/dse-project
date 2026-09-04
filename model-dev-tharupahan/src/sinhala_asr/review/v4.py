"""Build and finalize an audio-verified evaluation-reference revision."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from sinhala_asr.text.normalizer import canonicalize, metric_normalize


EVALUATION_SPLITS = ("validation", "test")


def select_v4_review_rows(
    rows: list[dict[str, Any]], *, control_rows: int = 100, seed: int = 20260904
) -> list[dict[str, Any]]:
    """Select every revised evaluation label plus deterministic unchanged controls."""
    evaluation = [row for row in rows if row.get("dataset_split") in EVALUATION_SPLITS]
    disputed = []
    unchanged = []
    for row in evaluation:
        normalized_original = canonicalize(str(row.get("text_original") or ""))
        item = dict(row)
        item["text_original"] = normalized_original
        item["previous_v3_transcript"] = str(row["text_canonical"])
        if normalized_original != str(row["text_canonical"]):
            item["review_category"] = f"disputed_{row['dataset_split']}"
            disputed.append(item)
        else:
            item["review_category"] = f"control_{row['dataset_split']}"
            unchanged.append(item)

    unchanged.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['sample_id']}".encode("utf-8")
        ).hexdigest()
    )
    controls = unchanged[:control_rows]
    selected = disputed + controls
    selected.sort(
        key=lambda row: (
            0 if str(row["review_category"]).startswith("disputed_") else 1,
            str(row["dataset_split"]),
            str(row["sample_id"]),
        )
    )
    if len({str(row["sample_id"]) for row in selected}) != len(selected):
        raise ValueError("v4 review selection contains duplicate sample IDs")
    return selected


def finalize_v4_rows(
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    adjudications: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Restore original evaluation labels and apply only complete audio decisions."""
    missing = selected_ids - set(adjudications)
    extra = set(adjudications) - selected_ids
    if missing or extra:
        raise ValueError(
            f"review coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    finalized = []
    applied_edits = 0
    excluded = 0
    heldout_unreviewed = 0
    for source in rows:
        row = dict(source)
        sample_id = str(row["sample_id"])
        if row.get("dataset_split") in EVALUATION_SPLITS:
            original = canonicalize(str(row.get("text_original") or ""))
            row["text_canonical"] = original
            row["text_metric"] = metric_normalize(original)
            if sample_id not in selected_ids:
                row["dataset_split"] = "heldout_unreviewed"
                row["exclusion_reason"] = "not_audio_reviewed_v4"
                heldout_unreviewed += 1
        if sample_id in selected_ids:
            record = adjudications[sample_id]
            decision = str(record["decision"])
            if decision == "edited":
                corrected = canonicalize(str(record.get("text_corrected") or ""))
                if not corrected or corrected == row["text_canonical"]:
                    raise ValueError(f"invalid edited transcript: {sample_id}")
                row["text_canonical"] = corrected
                row["text_metric"] = metric_normalize(corrected)
                applied_edits += 1
            elif decision == "correct":
                pass
            elif decision in {"bad_audio", "uncertain"}:
                row["dataset_split"] = "excluded"
                row["exclusion_reason"] = f"v4_audio_review_{decision}"
                excluded += 1
            else:
                raise ValueError(f"unsupported v4 decision for {sample_id}: {decision}")
        finalized.append(row)

    fingerprint_input = "\n".join(
        f"{row['sample_id']}\t{row['dataset_split']}\t{row.get('text_canonical', '')}"
        for row in sorted(finalized, key=lambda value: str(value["sample_id"]))
    )
    return finalized, {
        "split_counts": dict(sorted(Counter(r["dataset_split"] for r in finalized).items())),
        "reviewed_rows": len(selected_ids),
        "audio_verified_edits": applied_edits,
        "audio_review_exclusions": excluded,
        "heldout_unreviewed_rows": heldout_unreviewed,
        "dataset_fingerprint": hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest(),
    }
