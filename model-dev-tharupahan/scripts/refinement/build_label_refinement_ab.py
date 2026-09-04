#!/usr/bin/env python3
"""Build matched original/refined transcript manifests for quick A/B tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.text.normalizer import canonicalize, metric_normalize


LANGUAGES = ("sinhala_only", "latin_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_arms(
    manifest_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    refinement_rows: list[dict[str, Any]],
    *,
    accepted_confidence: set[str],
    validation_limits: dict[str, int | None],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_id = {str(row["sample_id"]): row for row in manifest_rows}
    if len(by_id) != len(manifest_rows):
        raise ValueError("manifest contains duplicate sample IDs")

    selected_by_language: dict[str, list[str]] = {key: [] for key in LANGUAGES}
    for selected in selected_rows:
        sample_id = str(selected["sample_id"])
        if sample_id not in by_id:
            raise ValueError(f"selected sample is absent from manifest: {sample_id}")
        source = by_id[sample_id]
        language = str(selected["language_class"])
        if language not in selected_by_language:
            raise ValueError(f"unsupported language class: {language}")
        if source.get("dataset_split") != "train":
            raise ValueError(f"selected non-training sample: {sample_id}")
        if source.get("language_class") != language:
            raise ValueError(f"language mismatch for selected sample: {sample_id}")
        if str(source["text_canonical"]) != str(selected["original"]):
            raise ValueError(f"source text changed since export: {sample_id}")
        selected_by_language[language].append(sample_id)

    refinements: dict[str, dict[str, Any]] = {}
    for result in refinement_rows:
        sample_id = str(result["sample_id"])
        if sample_id in refinements:
            raise ValueError(f"duplicate refinement result: {sample_id}")
        if sample_id not in by_id:
            raise ValueError(f"refinement sample is absent from manifest: {sample_id}")
        if str(result["original"]) != str(by_id[sample_id]["text_canonical"]):
            raise ValueError(f"refinement original does not match manifest: {sample_id}")
        refinements[sample_id] = result

    selected_ids = {sample_id for ids in selected_by_language.values() for sample_id in ids}
    if set(refinements) != selected_ids:
        missing = selected_ids - set(refinements)
        extra = set(refinements) - selected_ids
        raise ValueError(
            f"refinement coverage mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for language in LANGUAGES:
        original_train = [dict(by_id[sample_id]) for sample_id in selected_by_language[language]]
        refined_train = []
        for row in original_train:
            candidate = refinements[str(row["sample_id"])]
            refined = dict(row)
            if candidate.get("changed") and candidate.get("confidence") in accepted_confidence:
                corrected = canonicalize(str(candidate["corrected"]))
                if not corrected:
                    raise ValueError(f"blank accepted refinement: {row['sample_id']}")
                refined["text_canonical"] = corrected
                refined["text_metric"] = metric_normalize(corrected)
            refined_train.append(refined)

        validation = sorted(
            (
                dict(row)
                for row in manifest_rows
                if row.get("dataset_split") == "validation"
                and row.get("language_class") == language
            ),
            key=lambda row: str(row["sample_id"]),
        )
        limit = validation_limits[language]
        if limit is not None:
            validation = validation[:limit]
        if not original_train or not validation:
            raise ValueError(f"empty training or validation arm for {language}")
        output[language] = {
            "original": original_train + validation,
            "refined": refined_train + validation,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--refinements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confidence", action="append", choices=("high", "medium", "low"), default=[]
    )
    parser.add_argument("--sinhala-validation", type=int, default=100)
    parser.add_argument("--english-validation", type=int, default=36)
    args = parser.parse_args()
    manifest = args.manifest.expanduser().resolve()
    selection = args.selection.expanduser().resolve()
    refinements = args.refinements.expanduser().resolve()
    refinement_payload = json.loads(refinements.read_text(encoding="utf-8"))
    accepted = set(args.confidence or ["high", "medium"])
    arms = build_arms(
        pq.read_table(manifest).to_pylist(),
        json.loads(selection.read_text(encoding="utf-8")),
        refinement_payload["rows"],
        accepted_confidence=accepted,
        validation_limits={
            "sinhala_only": args.sinhala_validation,
            "latin_only": args.english_validation,
        },
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "selection_sha256": sha256(selection),
        "refinements_sha256": sha256(refinements),
        "refinement_model": refinement_payload.get("model"),
        "accepted_confidence": sorted(accepted),
        "languages": {},
    }
    refinement_by_id = refinement_payload_by_id(refinement_payload)
    for language, variants in arms.items():
        for variant, rows in variants.items():
            path = output / f"{language}-{variant}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        original_train = [r for r in variants["original"] if r["dataset_split"] == "train"]
        refined_train = [r for r in variants["refined"] if r["dataset_split"] == "train"]
        changed = [
            {
                "sample_id": before["sample_id"],
                "original": before["text_canonical"],
                "refined": after["text_canonical"],
                "confidence": refinement_by_id[str(before["sample_id"])]["confidence"],
            }
            for before, after in zip(original_train, refined_train)
            if before["text_canonical"] != after["text_canonical"]
        ]
        summary["languages"][language] = {
            "train_rows": len(original_train),
            "validation_rows": len(variants["original"]) - len(original_train),
            "accepted_changes": len(changed),
            "changes": changed,
        }
    (output / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["languages"], ensure_ascii=False, indent=2))


def refinement_payload_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in payload["rows"]}


if __name__ == "__main__":
    main()
