"""Lazy audio dataset backed by the immutable source Parquet files."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from torch.utils.data import Dataset

from sinhala_asr.data.manifest import _audio_bytes, _first_present


class ManifestAudioDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self._sources: dict[str, list[dict[str, Any]]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _source_row(self, row: dict[str, Any]) -> dict[str, Any]:
        path = str(row["source_path"])
        if path not in self._sources:
            self._sources[path] = pq.read_table(path).to_pylist()
        return self._sources[path][int(row["source_row_index"])]

    def __getitem__(self, index: int) -> dict[str, Any]:
        manifest_row = self.rows[index]
        source_path = Path(manifest_row["source_path"])
        source_row = self._source_row(manifest_row)
        raw = _audio_bytes(
            _first_present(source_row, "audio", "audio_path", "file"),
            source_path.parent,
        )
        samples, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        samples = np.mean(samples, axis=1)
        if sample_rate != 16000:
            raise ValueError(
                f"expected 16 kHz audio, found {sample_rate} Hz for {manifest_row['sample_id']}"
            )
        return {
            "audio": samples,
            "text": str(manifest_row["text_canonical"]),
            "sample_id": str(manifest_row["sample_id"]),
        }


def load_training_rows(
    manifest: Path, *, allow_unreviewed_validation: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = pq.read_table(manifest).to_pylist()
    train = [row for row in rows if row.get("dataset_split") == "train"]
    validation_split = (
        "validation_candidate" if allow_unreviewed_validation else "validation"
    )
    validation = [row for row in rows if row.get("dataset_split") == validation_split]
    if not train or not validation:
        raise ValueError(f"manifest must contain train and {validation_split} rows")
    return train, validation
