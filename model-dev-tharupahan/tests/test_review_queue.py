import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from sinhala_asr.data.manifest import build_manifest_rows, write_manifest
from sinhala_asr.review.queue import build_review_queue, select_review_rows


def wav_bytes(frequency: float) -> bytes:
    time = np.arange(1600, dtype=np.float32) / 16_000
    buffer = io.BytesIO()
    sf.write(buffer, 0.2 * np.sin(2 * np.pi * frequency * time), 16_000, format="WAV")
    return buffer.getvalue()


def test_selection_is_deterministic_unique_and_stratified():
    rows = [
        {"sample_id": "a", "source_dataset": "openslr52", "validation_flags": "[]", "is_code_switched": False, "audio_sha256": "1"},
        {"sample_id": "b", "source_dataset": "youtube", "validation_flags": "[]", "is_code_switched": True, "audio_sha256": "2"},
        {"sample_id": "c", "source_dataset": "bizbrains", "validation_flags": '["flag"]', "is_code_switched": False, "audio_sha256": "3"},
        {"sample_id": "d", "source_dataset": "linga", "validation_flags": "[]", "is_code_switched": False, "audio_sha256": "1"},
    ]
    first = select_review_rows(rows, quota=1, seed=7)
    second = select_review_rows(rows, quota=1, seed=7)
    assert first == second
    assert len({row["sample_id"] for row in first}) == len(first)
    assert {row["review_category"] for row in first} >= {"code_switched", "flagged_anomaly", "duplicate_candidate"}


def test_queue_embeds_audio_for_offline_review(tmp_path):
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"audio": wav_bytes(300), "text": "සිංහල text", "source_dataset": "youtube", "id": "x"}]
        ),
        source,
    )
    manifest = tmp_path / "manifest.parquet"
    write_manifest(build_manifest_rows("unsplit", source), manifest)
    output = tmp_path / "queue.parquet"
    queue = build_review_queue(manifest, output, quota=1)
    assert output.is_file()
    assert queue.iloc[0]["audio"].startswith(b"RIFF")
