import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from sinhala_asr.training.dataset import ManifestAudioDataset, load_crop_bounds


def test_manifest_audio_dataset_applies_non_destructive_crop(tmp_path: Path) -> None:
    audio = io.BytesIO()
    sf.write(audio, np.arange(16000, dtype=np.float32) / 16000, 16000, format="WAV")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([{"audio": audio.getvalue()}]), source)
    row = {
        "sample_id": "sample",
        "source_path": str(source),
        "source_row_index": 0,
        "text_canonical": "පාඨය",
    }
    original = ManifestAudioDataset([row])[0]
    cropped = ManifestAudioDataset([row], {"sample": (0.25, 0.75)})[0]
    assert len(original["audio"]) == 16000
    assert len(cropped["audio"]) == 8000
    assert cropped["text"] == original["text"]


def test_load_crop_bounds_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "crops.parquet"
    row = {"sample_id": "same", "crop_start_seconds": 0.1, "crop_end_seconds": 1.0}
    pq.write_table(pa.Table.from_pylist([row, row]), path)
    try:
        load_crop_bounds(path)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate crop IDs must be rejected")
