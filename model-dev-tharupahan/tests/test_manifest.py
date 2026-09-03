import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from sinhala_asr.data.audit import run_audit
from sinhala_asr.data.manifest import build_manifest_rows


def wav_bytes(*, duration=0.25, sample_rate=16_000, amplitude=0.2, frequency=440.0):
    time = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    audio = amplitude * np.sin(2 * np.pi * frequency * time)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_manifest_contains_stable_identity_audio_and_text_metadata(tmp_path):
    path = tmp_path / "train.parquet"
    write_parquet(
        path,
        [{"audio": wav_bytes(), "text": "  සිංහල  පාඨය .", "source_dataset": "fixture", "id": "a"}],
    )

    first = build_manifest_rows("train", path)[0]
    second = build_manifest_rows("train", path)[0]

    assert first["sample_id"] == second["sample_id"]
    assert len(first["audio_sha256"]) == 64
    assert len(first["audio_pcm_sha256"]) == 64
    assert first["text_original"] == "  සිංහල  පාඨය ."
    assert first["text_canonical"] == "සිංහල පාඨය."
    assert first["text_metric"] == "සිංහල පාඨය"
    assert first["sample_rate"] == 16_000
    assert first["channels"] == 1
    assert first["duration_seconds"] == 0.25
    assert first["is_valid"] is True


def test_invalid_rows_are_retained_with_reasons(tmp_path):
    path = tmp_path / "train.parquet"
    write_parquet(path, [{"audio": b"not audio", "text": "", "source_dataset": "fixture"}])

    row = build_manifest_rows("train", path)[0]
    flags = json.loads(row["validation_flags"])

    assert row["is_valid"] is False
    assert "invalid_audio" in flags
    assert "empty_transcript" in flags
    assert "empty_canonical_transcript" in flags
    assert row["audio_error"].startswith("LibsndfileError:")


def test_audit_detects_audio_and_sample_leakage_across_splits(tmp_path):
    audio = wav_bytes()
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    output = tmp_path / "report"
    row = {"audio": audio, "text": "එකම පාඨය", "source_dataset": "fixture"}
    write_parquet(train, [row])
    write_parquet(test, [row])

    summary = run_audit([("train", train), ("test", test)], output)

    assert summary["passed"] is False
    assert len(summary["audio_cross_split_leaks"]) == 1
    assert len(summary["sample_cross_split_leaks"]) == 1
    assert summary["duplicate_audio_hashes"] == 1
    assert summary["duplicate_pcm_audio_hashes"] == 1
    assert summary["total_audio_hours"] > 0
    assert (output / "manifest.parquet").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "summary.md").is_file()


def test_audit_detects_speaker_leakage(tmp_path):
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    write_parquet(
        train,
        [{"audio": wav_bytes(frequency=300), "text": "පුහුණු පාඨය", "source_dataset": "fixture", "speaker_id": "s1"}],
    )
    write_parquet(
        test,
        [{"audio": wav_bytes(frequency=600), "text": "පරීක්ෂණ පාඨය", "source_dataset": "fixture", "speaker_id": "s1"}],
    )

    summary = run_audit([("train", train), ("test", test)], tmp_path / "report")

    assert summary["speaker_audit_available"] is True
    assert len(summary["speaker_cross_split_leaks"]) == 1


def test_clean_disjoint_data_passes(tmp_path):
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    write_parquet(
        train,
        [{"audio": wav_bytes(frequency=300), "text": "පුහුණු පාඨය", "source_dataset": "fixture", "speaker_id": "s1"}],
    )
    write_parquet(
        test,
        [{"audio": wav_bytes(frequency=600), "text": "පරීක්ෂණ පාඨය", "source_dataset": "fixture", "speaker_id": "s2"}],
    )

    summary = run_audit([("train", train), ("test", test)], tmp_path / "report")

    assert summary["passed"] is True
    assert summary["invalid_rows"] == 0


def test_audit_detects_recording_group_leakage(tmp_path):
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    write_parquet(
        train,
        [{"audio": wav_bytes(frequency=300), "text": "පුහුණු", "source_dataset": "youtube", "video_id": "v1"}],
    )
    write_parquet(
        test,
        [{"audio": wav_bytes(frequency=600), "text": "පරීක්ෂණ", "source_dataset": "youtube", "video_id": "v1"}],
    )
    summary = run_audit([("train", train), ("test", test)], tmp_path / "report")
    assert summary["recording_group_audit_available"] is True
    assert len(summary["recording_group_cross_split_leaks"]) == 1
