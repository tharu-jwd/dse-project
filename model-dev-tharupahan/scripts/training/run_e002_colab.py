#!/usr/bin/env python3
"""Run resumable E002 Whisper-small wide-LoRA training on disposable Colab."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import tarfile
import time
import traceback
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
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL = "openai/whisper-small"
ROOT = Path("/content/sinhala-asr-job")
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
TRAIN_DIR = INPUT / "train"
VALIDATION_BUNDLE = INPUT / "v4-validation-206-audio.parquet"
JOB_CONFIG = INPUT / "job-config.json"
EVENTS = OUTPUT / "events.jsonl"
STATUS = OUTPUT / "status.json"
CHECKPOINT_INDEX = OUTPUT / "checkpoint-index.json"
FINAL_DIR = OUTPUT / "final-adapter"
PREDICTIONS = OUTPUT / "sinhala-validation-predictions.parquet"
METADATA = OUTPUT / "run-metadata.json"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def event(kind: str, **fields: object) -> None:
    record = {"time": time.time(), "event": kind, **fields}
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


class EmbeddedAudioDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], text_column: str) -> None:
        self.rows = rows
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
        if rate != 16000 or waveform.ndim != 1:
            raise ValueError(f"unexpected audio format: {row['sample_id']}")
        return {
            "audio": waveform,
            "text": row[self.text_column],
            "decoder_language": row.get("decoder_language") or "si",
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
        label_features = []
        for feature in features:
            self.processor.tokenizer.set_prefix_tokens(
                language=feature["decoder_language"], task="transcribe"
            )
            label_features.append(
                {"input_ids": self.processor.tokenizer(feature["text"]).input_ids}
            )
        encoded = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        self.processor.tokenizer.set_prefix_tokens(language="si", task="transcribe")
        labels = encoded.input_ids.masked_fill(encoded.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class DurableCheckpointCallback(TrainerCallback):
    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
        event("heartbeat", step=int(state.global_step), logs=logs or {})

    def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        step = int(state.global_step)
        checkpoint = Path(args.output_dir) / f"checkpoint-{step}"
        required = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
        ]
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise RuntimeError(f"incomplete checkpoint {step}: {missing}")
        (checkpoint / "COMPLETE").write_text(f"{step}\n", encoding="utf-8")
        archive = OUTPUT / f"checkpoint-{step:06d}.tar.gz"
        partial = archive.with_suffix(".tar.gz.partial")
        with tarfile.open(partial, "w:gz") as tar:
            tar.add(checkpoint, arcname=checkpoint.name)
        partial.replace(archive)
        record = {"step": step, "name": archive.name, "sha256": sha256(archive)}
        index = (
            json.loads(CHECKPOINT_INDEX.read_text())
            if CHECKPOINT_INDEX.exists()
            else {"checkpoints": []}
        )
        index["checkpoints"].append(record)
        write_json(CHECKPOINT_INDEX, index)
        event("checkpoint_complete", **record)


class StopAfterStepCallback(TrainerCallback):
    def __init__(self, stop_after_step: int | None) -> None:
        self.stop_after_step = stop_after_step

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        if self.stop_after_step is not None and state.global_step >= self.stop_after_step:
            control.should_save = True
            control.should_training_stop = True
            event("deliberate_stop", step=int(state.global_step))
        return control


def load_and_verify_inputs(config: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    manifest = json.loads((TRAIN_DIR / "manifest.json").read_text())
    if manifest["source_sha256"] != config["training_bundle_sha256"]:
        raise ValueError("training source bundle hash mismatch")
    tables = []
    for shard in manifest["shards"]:
        path = TRAIN_DIR / shard["name"]
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"training shard hash mismatch: {path.name}")
        tables.append(pq.read_table(path))
    train = pa.concat_tables(tables, promote_options="default").to_pylist()
    if len(train) != int(config["training_rows"]):
        raise ValueError("training row count mismatch")
    if sha256(VALIDATION_BUNDLE) != config["validation_bundle_sha256"]:
        raise ValueError("validation bundle hash mismatch")
    validation = pq.read_table(VALIDATION_BUNDLE).to_pylist()
    if len(validation) != 206:
        raise ValueError("validation row count mismatch")
    return train, validation


def generate_validation(
    model: Any,
    processor: Any,
    rows: list[dict],
    batch_size: int,
    experiment: str,
) -> float:
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    model.eval()
    predictions: list[str] = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            waveforms = []
            for row in batch:
                waveform, rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
                if rate != 16000 or waveform.ndim != 1:
                    raise ValueError(
                        f"unexpected validation audio: {row['sample_id']}"
                    )
                waveforms.append(waveform)
            inputs = processor.feature_extractor(
                waveforms,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            generated = model.generate(
                inputs.input_features.to("cuda", dtype=torch.float16),
                attention_mask=inputs.attention_mask.to("cuda"),
                max_new_tokens=64,
                no_repeat_ngram_size=3,
            )
            predictions.extend(
                text.strip()
                for text in processor.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
            )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {key: value for key, value in row.items() if key != "audio"}
                | {"prediction": prediction, "model": f"{MODEL}+{experiment}"}
                for row, prediction in zip(rows, predictions)
            ]
        ),
        PREDICTIONS,
        compression="zstd",
    )
    return time.monotonic() - started


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text('{"state":"preflight"}\n', encoding="utf-8")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        config = json.loads(JOB_CONFIG.read_text())
        write_json(OUTPUT / "resolved-config.json", config)
        train_rows, validation_rows = load_and_verify_inputs(config)
        event("preflight_complete", gpu=torch.cuda.get_device_name(0))

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
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=TARGET_MODULES,
            ),
        )
        model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if trainable != 6_488_064 or total != 248_222_976:
            raise RuntimeError(f"unexpected parameter counts: {trainable}/{total}")

        run_dir = OUTPUT / "trainer"
        arguments = Seq2SeqTrainingArguments(
            output_dir=str(run_dir),
            max_steps=int(config["max_steps"]),
            learning_rate=float(config["learning_rate"]),
            per_device_train_batch_size=int(config["train_batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            eval_strategy="no",
            save_strategy="steps",
            save_steps=int(config["save_steps"]),
            save_total_limit=2,
            logging_steps=int(config["logging_steps"]),
            fp16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            seed=int(config["seed"]),
            data_seed=int(config["seed"]),
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=arguments,
            train_dataset=EmbeddedAudioDataset(train_rows, "text"),
            data_collator=WhisperCollator(processor),
            processing_class=processor,
            callbacks=[
                DurableCheckpointCallback(),
                StopAfterStepCallback(config.get("stop_after_step")),
            ],
        )
        resume = config.get("resume_from_checkpoint")
        event("training_started", resume_from_checkpoint=resume)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = trainer.train(resume_from_checkpoint=resume)
        training_seconds = time.monotonic() - started
        trainer.save_model(FINAL_DIR)
        processor.save_pretrained(FINAL_DIR)
        evaluation_seconds = 0.0
        if not config.get("skip_validation", False):
            evaluation_seconds = generate_validation(
                model,
                processor,
                validation_rows,
                int(config["eval_batch_size"]),
                str(config.get("experiment", "e002")),
            )
        metadata = {
            "base_model": MODEL,
            "method": "wide_lora",
            "target_modules": TARGET_MODULES,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "train_metrics": result.metrics,
            "predictions_sha256": (
                sha256(PREDICTIONS) if PREDICTIONS.exists() else None
            ),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        }
        write_json(METADATA, metadata)
        write_json(STATUS, {"state": "complete", "exit_code": 0})
        event("job_completed")
    except Exception as error:
        failure = {
            "state": "failed",
            "exit_code": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(STATUS, failure)
        event("job_failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
