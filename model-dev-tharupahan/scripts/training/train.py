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
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from sinhala_asr.evaluation.metrics import score_pair, strict_normalize
from sinhala_asr.training.config import TrainConfig
from sinhala_asr.training.dataset import (
    ManifestAudioDataset,
    load_crop_bounds,
    load_training_rows,
)


@dataclass
class WhisperCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio = [feature["audio"] for feature in features]
        batch = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_attention_mask=True, return_tensors="pt"
        )
        label_features = []
        for feature in features:
            self.processor.tokenizer.set_prefix_tokens(
                language=feature.get("decoder_language") or "si", task="transcribe"
            )
            label_features.append(
                {"input_ids": self.processor.tokenizer(feature["text"]).input_ids}
            )
        label_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        # Restore the inference/default prompt after constructing a mixed-
        # language batch. Prefix selection above is deliberately per row.
        self.processor.tokenizer.set_prefix_tokens(language="si", task="transcribe")
        labels = label_batch.input_ids.masked_fill(
            label_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class StopAfterStepCallback(TrainerCallback):
    """Create a normal resumable checkpoint and stop at a phase boundary."""

    def __init__(self, stop_after_step: int | None) -> None:
        self.stop_after_step = stop_after_step

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        if self.stop_after_step is not None and state.global_step >= self.stop_after_step:
            control.should_save = True
            control.should_training_stop = True
        return control


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    # Record-keeping only, not load-bearing for the run itself -- degrade to
    # "unknown" rather than crashing when there is no git checkout at all
    # (e.g. a Camber job populated from a stash mount, not a git clone).
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-train-rows", type=int)
    parser.add_argument("--smoke-validation-rows", type=int)
    parser.add_argument("--allow-unreviewed-validation", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = TrainConfig.load(config_path)
    manifest = Path(config.manifest).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows = load_training_rows(
        manifest, allow_unreviewed_validation=args.allow_unreviewed_validation
    )
    crop_bounds = None
    if config.crop_training_audio:
        crop_bounds = load_crop_bounds(
            Path(str(config.crop_proposals)).expanduser().resolve()
        )
        missing = {str(row["sample_id"]) for row in train_rows} - set(crop_bounds)
        if missing:
            raise ValueError(f"crop proposals missing {len(missing)} training samples")
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
    if config.crop_training_audio:
        metadata["crop_proposals_sha256"] = fingerprint(
            Path(str(config.crop_proposals)).expanduser().resolve()
        )
    (output_dir / "run-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    processor = WhisperProcessor.from_pretrained(
        config.model_name, language="si", task="transcribe"
    )
    if config.extended_tokenizer_path:
        # Swap in a tokenizer with extra Sinhala subword tokens appended via
        # add_tokens() (see scripts/training/extend_tokenizer.py); the base
        # tokenizer's own vocab/merges are untouched, only new IDs are added
        # on top, so this is purely additive.
        processor.tokenizer = WhisperTokenizer.from_pretrained(
            config.extended_tokenizer_path, language="si", task="transcribe"
        )
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    if config.extended_tokenizer_path:
        # New rows get the model's default (random) initialization -- the
        # same approach used, without reported instability, by the external
        # precedent for this exact technique (see
        # docs/audits/e006-near-homophone-error-analysis.md#external-precedent-for-this-exact-fix).
        model.resize_token_embeddings(len(processor.tokenizer))
    model.generation_config.language = "si"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    # Without this, eval-time greedy decoding on an under-trained checkpoint
    # (e.g. an early Optuna trial capped at 100 steps) can degenerate into an
    # infinite repeated-token loop that runs to generation_max_length, which
    # inflates eval_cer past 100% and pins eval_wer at ~1.0 regardless of the
    # hyperparameters under test. Confirmed by direct reproduction: identical
    # checkpoint decoded with and without this set (see
    # docs/audits/e008-eval-repetition-bug.md).
    model.generation_config.no_repeat_ngram_size = 3
    if config.method == "lora":
        from peft import LoraConfig, get_peft_model

        # Whisper ties its input embeddings and output projection
        # (model.decoder.embed_tokens / proj_out share one weight matrix).
        # Neither is in target_modules above, so get_peft_model() freezes
        # them by default -- fine normally, but when extended_tokenizer_path
        # has just added new, randomly-initialized rows to that matrix via
        # resize_token_embeddings(), leaving them frozen means they can
        # never learn anything: confirmed directly to cause severe training
        # instability (the model falling back on unrelated tokens from its
        # original pretraining, including a different script entirely) --
        # see docs/audits/e009-tokenizer-extension.md. modules_to_save keeps
        # them fully trainable (not LoRA-adapted, plain fine-tuned)
        # alongside the adapters, the standard PEFT pattern for vocabulary
        # extension.
        modules_to_save = ["embed_tokens", "proj_out"] if config.extended_tokenizer_path else None
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                modules_to_save=modules_to_save,
                # Without this, PEFT creates two independent trainable
                # copies of the tied embed_tokens/proj_out matrix instead
                # of keeping them tied -- confirmed directly (verified via
                # named_parameters()): with ensure_weight_tying=True there
                # is exactly one shared trainable copy, matching the base
                # model's own tie_word_embeddings=True architecture.
                ensure_weight_tying=bool(config.extended_tokenizer_path),
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
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
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
        neftune_noise_alpha=config.neftune_noise_alpha,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ManifestAudioDataset(train_rows, crop_bounds=crop_bounds),
        eval_dataset=ManifestAudioDataset(validation_rows),
        data_collator=WhisperCollator(processor),
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[StopAfterStepCallback(config.stop_after_step)],
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
