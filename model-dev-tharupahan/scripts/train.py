#!/usr/bin/env python3
"""Run a budget-gated, resumable Whisper full or LoRA fine-tune."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from sinhala_asr.evaluation.metrics import score_pair, strict_normalize
from sinhala_asr.training.config import TrainConfig
from sinhala_asr.training.dataset import ManifestAudioDataset, load_training_rows


@dataclass
class WhisperCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [feature["audio"] for feature in features]
        batch = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_attention_mask=True, return_tensors="pt"
        )
        label_features = self.processor.tokenizer(
            [feature["text"] for feature in features], padding=True, return_tensors="pt"
        )
        labels = label_features.input_ids.masked_fill(
            label_features.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-train-rows", type=int)
    parser.add_argument("--smoke-validation-rows", type=int)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = TrainConfig.load(config_path)
    manifest = Path(config.manifest).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows = load_training_rows(
        manifest, allow_unreviewed_validation=args.smoke_validation_rows is not None
    )
    if args.smoke_train_rows:
        train_rows = train_rows[: args.smoke_train_rows]
    if args.smoke_validation_rows:
        validation_rows = validation_rows[: args.smoke_validation_rows]

    metadata = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "resolved_config": config.resolved(),
        "manifest_path": str(manifest),
        "manifest_sha256": fingerprint(manifest),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    processor = WhisperProcessor.from_pretrained(
        config.model_name, language="si", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    model.generation_config.language = "si"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    if config.method == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            ),
        )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        prediction_ids = prediction.predictions
        label_ids = np.where(
            prediction.label_ids == -100,
            processor.tokenizer.pad_token_id,
            prediction.label_ids,
        )
        hypotheses = processor.tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
        references = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        scores = [
            score_pair(reference, hypothesis, strict_normalize)
            for reference, hypothesis in zip(references, hypotheses)
        ]
        word_errors = sum(score["word_errors"] for score in scores)
        word_units = sum(score["word_reference_units"] for score in scores)
        character_errors = sum(score["character_errors"] for score in scores)
        character_units = sum(score["character_reference_units"] for score in scores)
        return {
            "wer": word_errors / word_units if word_units else 0.0,
            "cer": character_errors / character_units if character_units else 0.0,
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        logging_steps=config.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        load_best_model_at_end=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ManifestAudioDataset(train_rows),
        eval_dataset=ManifestAudioDataset(validation_rows),
        data_collator=WhisperCollator(processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    actual_hours = (time.monotonic() - started) / 3600
    metadata["actual_training_hours"] = actual_hours
    metadata["actual_compute_cost_usd"] = actual_hours * config.hourly_price_usd
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
