#!/usr/bin/env python3
"""Run the bounded Whisper-small wide-LoRA pilot inside Google Colab."""

from __future__ import annotations

import io
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import peft
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL = "openai/whisper-small"
TRAIN_BUNDLE = Path("/content/v4-training-pilot-2000.parquet")
VALIDATION_BUNDLE = Path("/content/v4-validation-colab.parquet")
DRIVE_ROOT = Path("/content/drive/MyDrive/sinhala-asr/wide-lora-100-v1")
FINAL_DIR = DRIVE_ROOT / "final-adapter"
PREDICTIONS = Path("/content/wide-lora-100-v4-validation-predictions.parquet")
METADATA = Path("/content/wide-lora-100-run-metadata.json")

MAX_STEPS = 100
TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
GENERATION_MAX_NEW_TOKENS = 64
NO_REPEAT_NGRAM_SIZE = 3
SEED = 20260903


class EmbeddedAudioDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], text_column: str) -> None:
        self.rows = rows
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
        if sample_rate != 16000 or waveform.ndim != 1:
            raise ValueError(f"unexpected audio format for {row['sample_id']}")
        return {
            "audio": waveform,
            "text": row[self.text_column],
            "sample_id": row["sample_id"],
        }


@dataclass
class WhisperCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor(
            [feature["audio"] for feature in features],
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = self.processor.tokenizer(
            [feature["text"] for feature in features],
            padding=True,
            return_tensors="pt",
        )
        labels = encoded.input_ids.masked_fill(encoded.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required; select a Colab T4 runtime")
    for path in (TRAIN_BUNDLE, VALIDATION_BUNDLE):
        if not path.exists():
            raise SystemExit(f"missing uploaded bundle: {path}")
    if DRIVE_ROOT.exists() and any(DRIVE_ROOT.iterdir()):
        raise SystemExit(f"refusing to overwrite existing run: {DRIVE_ROOT}")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

    train_rows = pq.read_table(TRAIN_BUNDLE).to_pylist()
    validation_rows = pq.read_table(VALIDATION_BUNDLE).to_pylist()
    if len(train_rows) != 2000 or len(validation_rows) != 206:
        raise SystemExit("unexpected train/validation row count")

    processor = WhisperProcessor.from_pretrained(
        MODEL, language="si", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    model.generation_config.language = "si"
    model.generation_config.task = "transcribe"
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
        ),
    )
    model.enable_input_require_grads()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable >= total:
        raise SystemExit("base model was not frozen")
    model.print_trainable_parameters()

    arguments = Seq2SeqTrainingArguments(
        output_dir=str(DRIVE_ROOT),
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=5,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=EmbeddedAudioDataset(train_rows, "text"),
        data_collator=WhisperCollator(processor),
        processing_class=processor,
    )

    torch.cuda.reset_peak_memory_stats()
    training_started = time.monotonic()
    train_result = trainer.train()
    training_seconds = time.monotonic() - training_started
    trainer.save_model(FINAL_DIR)
    processor.save_pretrained(FINAL_DIR)

    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    model.eval()
    predictions: list[str] = []
    evaluation_started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(validation_rows), 8):
            rows = validation_rows[start : start + 8]
            audio = []
            for row in rows:
                waveform, rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
                if rate != 16000 or waveform.ndim != 1:
                    raise ValueError(f"unexpected validation audio: {row['sample_id']}")
                audio.append(waveform)
            features = processor.feature_extractor(
                audio,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            generated = model.generate(
                features.input_features.to("cuda", dtype=torch.float16),
                attention_mask=features.attention_mask.to("cuda"),
                max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            )
            predictions.extend(
                processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
            print(f"validation {min(start + 8, len(validation_rows))}/{len(validation_rows)}")
    evaluation_seconds = time.monotonic() - evaluation_started

    output_rows = []
    for row, prediction in zip(validation_rows, predictions):
        output_rows.append(
            {key: value for key, value in row.items() if key != "audio"}
            | {"prediction": prediction.strip(), "model": f"{MODEL}+wide-lora-100"}
        )
    pq.write_table(pa.Table.from_pylist(output_rows), PREDICTIONS, compression="zstd")
    metadata = {
        "base_model": MODEL,
        "method": "wide_lora",
        "target_modules": TARGET_MODULES,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "max_steps": MAX_STEPS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "train_metrics": train_result.metrics,
        "generation_max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "adapter_drive_path": str(FINAL_DIR),
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (DRIVE_ROOT / METADATA.name).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
