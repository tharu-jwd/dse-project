#!/usr/bin/env python3
"""Generate metadata-preserving Whisper predictions from a frozen manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from sinhala_asr.evaluation.predict import select_prediction_rows
from sinhala_asr.training.dataset import ManifestAudioDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--unlock-test", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.max_rows is not None and args.max_rows <= 0):
        raise SystemExit("batch-size and max-rows must be positive")
    manifest = args.manifest.expanduser().resolve()
    rows = pq.read_table(manifest).to_pylist()
    selected = select_prediction_rows(
        rows,
        args.split,
        allow_unreviewed=args.max_rows is not None,
        unlock_test=args.unlock_test,
    )
    if args.max_rows is not None:
        selected = selected[: args.max_rows]

    processor = WhisperProcessor.from_pretrained(
        args.model, language="si", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device).eval()
    model.generation_config.language = "si"
    model.generation_config.task = "transcribe"
    dataset = ManifestAudioDataset(selected)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    predictions: list[str] = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            features = processor.feature_extractor(
                [item["audio"] for item in batch],
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            generated = model.generate(
                features.input_features.to(device),
                attention_mask=features.attention_mask.to(device),
            )
            predictions.extend(
                processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )

    output_rows = []
    for row, prediction in zip(selected, predictions):
        output_rows.append(
            dict(row)
            | {
                "reference": str(row.get("text_reviewed") or row["text_canonical"]),
                "prediction": prediction.strip(),
                "model": args.model,
            }
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), output, compression="zstd")
    metadata = {
        "model": args.model,
        "manifest": str(manifest),
        "split": args.split,
        "rows": len(output_rows),
        "device": str(device),
        "runtime_seconds": time.monotonic() - started,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
