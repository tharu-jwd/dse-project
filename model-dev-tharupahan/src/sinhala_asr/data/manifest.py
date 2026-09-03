"""Build deterministic, metadata-rich manifests from ASR Parquet files."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from sinhala_asr.text.normalizer import (
    NORMALIZATION_VERSION,
    canonicalize,
    metric_normalize,
    transcript_flags,
)

MANIFEST_VERSION = "manifest-v2"


def _digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _first_present(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return None


def _audio_bytes(value: Any, base_dir: Path) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"])
        if value.get("path"):
            path = Path(value["path"])
            return (path if path.is_absolute() else base_dir / path).read_bytes()
    if isinstance(value, str):
        path = Path(value)
        return (path if path.is_absolute() else base_dir / path).read_bytes()
    raise TypeError(f"unsupported audio value: {type(value).__name__}")


def _audio_metadata(raw: bytes, silence_rms: float, clipping_peak: float) -> dict[str, Any]:
    with sf.SoundFile(io.BytesIO(raw)) as handle:
        sample_rate = int(handle.samplerate)
        channels = int(handle.channels)
        frames = int(handle.frames)
        audio_format = str(handle.format)
        subtype = str(handle.subtype)
        samples = handle.read(dtype="float32", always_2d=True)

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    pcm_identity = sample_rate.to_bytes(4, "little") + channels.to_bytes(2, "little") + samples.tobytes()
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_seconds": frames / sample_rate if sample_rate else 0.0,
        "audio_format": audio_format,
        "audio_subtype": subtype,
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "audio_pcm_sha256": _digest(pcm_identity),
        "is_silent": rms < silence_rms,
        "is_clipped": peak >= clipping_peak,
        "audio_error": None,
    }


def iter_parquet_rows(path: Path, batch_size: int = 512) -> Iterator[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row_index, row
            row_index += 1


def build_manifest_rows(
    split: str,
    path: Path,
    *,
    min_duration: float = 0.1,
    max_duration: float = 30.0,
    silence_rms: float = 1e-4,
    clipping_peak: float = 0.999,
) -> list[dict[str, Any]]:
    """Build manifest rows without silently dropping invalid samples."""
    rows: list[dict[str, Any]] = []
    for row_index, source_row in iter_parquet_rows(path):
        original = _first_present(source_row, "text", "transcription", "Transcription")
        known_sources = {"openslr52", "youtube", "bizbrains", "linga"}
        inferred_source = next((part for part in reversed(path.parts) if part in known_sources), "unknown")
        source = str(source_row.get("source_dataset") or inferred_source)
        source_record_id = (
            source_row.get("source_record_id")
            or source_row.get("id")
            or source_row.get("file_id")
            or source_row.get("FileID")
        )
        speaker_id = (
            source_row.get("speaker_id")
            or source_row.get("speaker")
            or source_row.get("Speaker")
        )
        recording_group_id = _first_present(
            source_row, "recording_group_id", "recording_id", "session_id", "video_id"
        )
        flags = transcript_flags(original)
        canonical = canonicalize(original) if isinstance(original, str) else ""
        metric_text = metric_normalize(original) if isinstance(original, str) else ""
        if not canonical:
            flags.append("empty_canonical_transcript")

        audio_hash = ""
        metadata: dict[str, Any] = {
            "sample_rate": None,
            "channels": None,
            "frames": None,
            "duration_seconds": None,
            "audio_format": None,
            "audio_subtype": None,
            "peak_amplitude": None,
            "rms_amplitude": None,
            "audio_pcm_sha256": "",
            "is_silent": None,
            "is_clipped": None,
            "audio_error": None,
        }
        try:
            audio_value = _first_present(source_row, "audio", "audio_path", "file")
            raw = _audio_bytes(audio_value, path.parent)
            audio_hash = _digest(raw)
            metadata = _audio_metadata(raw, silence_rms, clipping_peak)
            if metadata["duration_seconds"] < min_duration:
                flags.append("audio_too_short")
            if metadata["duration_seconds"] > max_duration:
                flags.append("audio_too_long")
            if metadata["is_silent"]:
                flags.append("silent_audio")
            if metadata["is_clipped"]:
                flags.append("clipped_audio")
        except Exception as error:  # retained in manifest for investigation
            flags.append("invalid_audio")
            metadata["audio_error"] = f"{type(error).__name__}: {error}"

        transcript_hash = _digest(canonical)
        identity = str(source_record_id) if source_record_id is not None else f"{audio_hash}:{transcript_hash}"
        sample_id = _digest(f"{source}:{identity}")[:24]
        blocking_flags = {
            "non_string_transcript",
            "empty_transcript",
            "empty_canonical_transcript",
            "invalid_audio",
            "audio_too_short",
            "audio_too_long",
            "silent_audio",
        }
        unique_flags = sorted(set(flags))
        if "code_switched_latin" in unique_flags:
            language_class = "code_switched"
        elif "latin_only" in unique_flags:
            language_class = "latin_only"
        elif any("\u0d80" <= char <= "\u0dff" for char in canonical):
            language_class = "sinhala_only"
        else:
            language_class = "other"
        rows.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "sample_id": sample_id,
                "split": split,
                "source_dataset": source,
                "source_record_id": None if source_record_id is None else str(source_record_id),
                "speaker_id": None if speaker_id is None else str(speaker_id),
                "speaker_key": None if speaker_id is None else f"{source}:{speaker_id}",
                "recording_group_id": None if recording_group_id is None else str(recording_group_id),
                "recording_group_key": (
                    None if recording_group_id is None else f"{source}:{recording_group_id}"
                ),
                "domain": source_row.get("domain"),
                "uploader": source_row.get("uploader"),
                "speaker_gender": source_row.get("speaker_gender"),
                "source_has_noise": source_row.get("has_noise"),
                "source_path": str(path),
                "source_row_index": row_index,
                "audio_sha256": audio_hash,
                "transcript_sha256": transcript_hash,
                "text_original": original if isinstance(original, str) else None,
                "text_canonical": canonical,
                "text_metric": metric_text,
                "is_code_switched": "code_switched_latin" in unique_flags,
                "language_class": language_class,
                "validation_flags": json.dumps(unique_flags, ensure_ascii=False),
                "is_valid": not bool(blocking_flags.intersection(unique_flags)),
                **metadata,
            }
        )
    return rows


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
