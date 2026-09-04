"""Validated, auditable training configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    manifest: str
    output_dir: str
    method: str = "full"
    max_steps: int = 1000
    learning_rate: float = 1e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    seed: int = 20260903
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    resume_from_checkpoint: str | None = None
    hourly_price_usd: float = 0.0
    estimated_hours: float = 0.0
    maximum_cost_usd: float = 0.0
    safety_margin: float = 1.25
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    crop_training_audio: bool = False
    crop_proposals: str | None = None

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        values: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.method not in {"full", "lora"}:
            raise ValueError("method must be 'full' or 'lora'")
        for name in (
            "max_steps",
            "train_batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be enabled")
        if self.crop_training_audio and not self.crop_proposals:
            raise ValueError("crop_training_audio requires crop_proposals")
        if min(self.hourly_price_usd, self.estimated_hours, self.maximum_cost_usd) < 0:
            raise ValueError("cost fields cannot be negative")
        if self.planned_cost_usd > self.maximum_cost_usd:
            raise ValueError(
                f"planned cost ${self.planned_cost_usd:.2f} exceeds maximum "
                f"${self.maximum_cost_usd:.2f}"
            )

    @property
    def planned_cost_usd(self) -> float:
        return self.hourly_price_usd * self.estimated_hours * self.safety_margin

    def resolved(self) -> dict[str, Any]:
        return asdict(self) | {"planned_cost_usd": self.planned_cost_usd}
