import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sinhala_asr.training.config import TrainConfig
from sinhala_asr.training.dataset import load_training_rows


def test_cost_gate_rejects_run_above_cap(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "model",
                "manifest": "manifest.parquet",
                "output_dir": "run",
                "hourly_price_usd": 1.0,
                "estimated_hours": 10,
                "maximum_cost_usd": 10,
            }
        )
    )
    with pytest.raises(ValueError, match="planned cost"):
        TrainConfig.load(path)


def test_cost_gate_uses_safety_margin() -> None:
    config = TrainConfig(
        "model",
        "manifest",
        "run",
        hourly_price_usd=0.4,
        estimated_hours=2,
        maximum_cost_usd=1,
    )
    config.validate()
    assert config.planned_cost_usd == 1.0


def test_unreviewed_validation_is_smoke_only(tmp_path: Path) -> None:
    path = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"sample_id": "train", "dataset_split": "train"},
                {"sample_id": "candidate", "dataset_split": "validation_candidate"},
            ]
        ),
        path,
    )
    with pytest.raises(ValueError, match="validation rows"):
        load_training_rows(path)
    train, validation = load_training_rows(path, allow_unreviewed_validation=True)
    assert train[0]["sample_id"] == "train"
    assert validation[0]["sample_id"] == "candidate"
