"""CLI and reporting for dataset integrity, duplication, and leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .manifest import MANIFEST_VERSION, build_manifest_rows, write_manifest
from sinhala_asr.text.normalizer import NORMALIZATION_VERSION


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be SPLIT=PATH")
    split, raw_path = value.split("=", 1)
    if not split.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("input must contain a split and path")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Parquet file does not exist: {path}")
    return split.strip(), path


def _cross_split(values: dict[str, set[str]]) -> list[dict[str, Any]]:
    occurrences: dict[str, set[str]] = defaultdict(set)
    for split, split_values in values.items():
        for value in split_values:
            if value:
                occurrences[value].add(split)
    return [
        {"value": value, "splits": sorted(splits)}
        for value, splits in sorted(occurrences.items())
        if len(splits) > 1
    ]


def summarize(rows: list[dict[str, Any]], inputs: list[tuple[str, Path]]) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    source_counts = Counter(row["source_dataset"] for row in rows)
    flag_counts: Counter[str] = Counter()
    for row in rows:
        flag_counts.update(json.loads(row["validation_flags"]))

    audio_by_split: dict[str, set[str]] = defaultdict(set)
    pcm_audio_by_split: dict[str, set[str]] = defaultdict(set)
    sample_by_split: dict[str, set[str]] = defaultdict(set)
    speaker_by_split: dict[str, set[str]] = defaultdict(set)
    recording_group_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        audio_by_split[row["split"]].add(row["audio_sha256"])
        pcm_audio_by_split[row["split"]].add(row["audio_pcm_sha256"])
        sample_by_split[row["split"]].add(row["sample_id"])
        if row["speaker_key"]:
            speaker_by_split[row["split"]].add(row["speaker_key"])
        if row["recording_group_key"]:
            recording_group_by_split[row["split"]].add(row["recording_group_key"])

    audio_leaks = _cross_split(audio_by_split)
    pcm_audio_leaks = _cross_split(pcm_audio_by_split)
    sample_leaks = _cross_split(sample_by_split)
    speaker_leaks = _cross_split(speaker_by_split)
    recording_group_leaks = _cross_split(recording_group_by_split)
    audio_counts = Counter(row["audio_sha256"] for row in rows if row["audio_sha256"])
    pcm_audio_counts = Counter(row["audio_pcm_sha256"] for row in rows if row["audio_pcm_sha256"])
    transcript_counts = Counter(row["transcript_sha256"] for row in rows if row["text_canonical"])
    invalid_rows = sum(not row["is_valid"] for row in rows)
    durations = sorted(float(row["duration_seconds"]) for row in rows if row["duration_seconds"] is not None)

    def percentile(fraction: float) -> float | None:
        if not durations:
            return None
        return durations[round((len(durations) - 1) * fraction)]
    fingerprint_lines = [
        f"{row['sample_id']}\t{row['split']}\t{row['audio_sha256']}\t{row['transcript_sha256']}"
        for row in sorted(rows, key=lambda item: (item["sample_id"], item["split"]))
    ]
    fingerprint = hashlib.sha256("\n".join(fingerprint_lines).encode()).hexdigest()

    violations: list[str] = []
    if invalid_rows:
        violations.append(f"{invalid_rows} invalid rows")
    duplicate_audio_hashes = sum(count > 1 for count in audio_counts.values())
    duplicate_pcm_audio_hashes = sum(count > 1 for count in pcm_audio_counts.values())
    if duplicate_audio_hashes:
        violations.append(f"{duplicate_audio_hashes} duplicate audio hashes")
    if audio_leaks:
        violations.append(f"{len(audio_leaks)} audio hashes cross split boundaries")
    if pcm_audio_leaks:
        violations.append(f"{len(pcm_audio_leaks)} decoded audio hashes cross split boundaries")
    if sample_leaks:
        violations.append(f"{len(sample_leaks)} sample IDs cross split boundaries")
    if speaker_leaks:
        violations.append(f"{len(speaker_leaks)} speaker IDs cross split boundaries")
    if recording_group_leaks:
        violations.append(f"{len(recording_group_leaks)} recording groups cross split boundaries")

    return {
        "manifest_version": MANIFEST_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "dataset_fingerprint": fingerprint,
        "inputs": [{"split": split, "path": str(path)} for split, path in inputs],
        "total_rows": len(rows),
        "valid_rows": len(rows) - invalid_rows,
        "invalid_rows": invalid_rows,
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "duplicate_audio_hashes": duplicate_audio_hashes,
        "duplicate_pcm_audio_hashes": duplicate_pcm_audio_hashes,
        "repeated_transcript_hashes": sum(count > 1 for count in transcript_counts.values()),
        "total_audio_hours": sum(durations) / 3600,
        "mean_duration_seconds": mean(durations) if durations else None,
        "duration_percentiles_seconds": {
            "p01": percentile(0.01),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        },
        "code_switched_rows": sum(bool(row["is_code_switched"]) for row in rows),
        "audio_cross_split_leaks": audio_leaks,
        "decoded_audio_cross_split_leaks": pcm_audio_leaks,
        "sample_cross_split_leaks": sample_leaks,
        "speaker_cross_split_leaks": speaker_leaks,
        "recording_group_cross_split_leaks": recording_group_leaks,
        "speaker_audit_available": bool(speaker_by_split),
        "recording_group_audit_available": bool(recording_group_by_split),
        "violations": violations,
        "passed": not violations,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    status = "PASS" if summary["passed"] else "FAIL"
    lines = [
        "# Dataset Audit",
        "",
        f"**Status:** {status}",
        "",
        f"- Dataset fingerprint: `{summary['dataset_fingerprint']}`",
        f"- Rows: {summary['total_rows']}",
        f"- Valid rows: {summary['valid_rows']}",
        f"- Invalid rows: {summary['invalid_rows']}",
        f"- Duplicate audio hashes: {summary['duplicate_audio_hashes']}",
        f"- Duplicate decoded-audio hashes: {summary['duplicate_pcm_audio_hashes']}",
        f"- Repeated transcript hashes: {summary['repeated_transcript_hashes']}",
        f"- Total audio: {summary['total_audio_hours']:.2f} hours",
        f"- Code-switched rows: {summary['code_switched_rows']}",
        f"- Speaker audit available: {summary['speaker_audit_available']}",
        f"- Recording-group audit available: {summary['recording_group_audit_available']}",
        "",
        "## Split counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary["split_counts"].items())
    lines.extend(["", "## Validation flags", ""])
    if summary["flag_counts"]:
        lines.extend(f"- {key}: {value}" for key, value in summary["flag_counts"].items())
    else:
        lines.append("- None")
    lines.extend(["", "## Violations", ""])
    lines.extend(f"- {value}" for value in summary["violations"] or ["None"])
    return "\n".join(lines) + "\n"


def run_audit(inputs: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split, path in inputs:
        rows.extend(build_manifest_rows(split, path))
    summary = summarize(rows, inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, output_dir / "manifest.parquet")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True, metavar="SPLIT=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()
    summary = run_audit(args.input, args.output_dir.expanduser().resolve())
    print(render_markdown(summary))
    if not summary["passed"] and not args.allow_invalid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
