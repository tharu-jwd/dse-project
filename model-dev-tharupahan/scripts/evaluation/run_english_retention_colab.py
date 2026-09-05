#!/usr/bin/env python3
"""Evaluate untouched Whisper-small and an optional LoRA adapter on English."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path

import peft
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import transformers
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = os.environ.get("SINHALA_ASR_MODEL", "openai/whisper-small")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(rows: list[dict], adapter: Path | None, batch_size: int) -> tuple[list[str], float]:
    processor = WhisperProcessor.from_pretrained(MODEL, language="en", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    model.to("cuda").eval()
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    predictions: list[str] = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            waveforms = []
            for row in batch:
                waveform, rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
                if rate != 16000 or waveform.ndim != 1:
                    raise ValueError(f"unexpected audio format: {row['sample_id']}")
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
                max_new_tokens=128,
                no_repeat_ngram_size=3,
            )
            predictions.extend(
                text.strip()
                for text in processor.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
            )
            print(f"predicted {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return predictions, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--variant", choices=("untouched", "e001", "e002", "e003"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if (args.variant != "untouched") != (args.adapter is not None):
        raise SystemExit("adapter variants require --adapter; untouched must omit it")

    rows = pq.read_table(args.benchmark).to_pylist()
    if len(rows) != 2620:
        raise SystemExit(f"unexpected English benchmark size: {len(rows)}")
    predictions, runtime = predict(rows, args.adapter, args.batch_size)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.variant}-predictions.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {key: value for key, value in row.items() if key != "audio"}
                | {"prediction": prediction, "model": f"{MODEL}:{args.variant}"}
                for row, prediction in zip(rows, predictions)
            ]
        ),
        output,
        compression="zstd",
    )
    metadata = {
        "base_model": MODEL,
        "variant": args.variant,
        "adapter": str(args.adapter) if args.adapter else None,
        "benchmark_sha256": file_hash(args.benchmark),
        "predictions_sha256": file_hash(output),
        "rows": len(rows),
        "batch_size": args.batch_size,
        "runtime_seconds": runtime,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    (output_dir / f"{args.variant}-runtime.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
