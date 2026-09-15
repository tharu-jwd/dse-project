#!/usr/bin/env python3
"""Evaluate Yohan's pinned run7 checkpoint on v6 validation and English."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "e016-yohan-run7-evaluation"
REPO = "Yohan2003/whisper-small-sinhala"
REVISION = "bdaa42b21afa3923e84e6a98d54cfc984c4c99c3"
SUBFOLDER = "models/run7-v4-lr3e-5-bs64"


def one_file(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_audio(blob: bytes) -> list[float]:
    waveform, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    if rate != 16000:
        raise RuntimeError(f"unexpected sample rate: {rate}")
    return waveform.mean(axis=1)


def predict(
    model,
    processor,
    rows: list[dict],
    audio_blobs: list[bytes],
    *,
    language: str,
    batch_size: int,
    num_beams: int,
) -> tuple[list[str], float]:
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    predictions: list[str] = []
    started = time.monotonic()
    for start in range(0, len(rows), batch_size):
        waveforms = [decode_audio(blob) for blob in audio_blobs[start : start + batch_size]]
        inputs = processor.feature_extractor(
            waveforms,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            generated = model.generate(
                inputs.input_features.to("cuda", dtype=torch.float16),
                attention_mask=inputs.attention_mask.to("cuda"),
                max_new_tokens=128,
                no_repeat_ngram_size=3,
                num_beams=num_beams,
            )
        predictions.extend(
            text.strip()
            for text in processor.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
        )
        print(f"{language}: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return predictions, time.monotonic() - started


def write_predictions(path: Path, rows: list[dict], predictions: list[str]) -> None:
    output_rows = []
    for row, prediction in zip(rows, predictions):
        clean = {key: value for key, value in row.items() if key != "audio"}
        clean["reference"] = str(
            row.get("reference")
            or row.get("text_reviewed")
            or row.get("text_canonical")
            or row["text"]
        )
        clean["prediction"] = prediction
        clean["model"] = f"{REPO}@{REVISION}:{SUBFOLDER}"
        output_rows.append(clean)
    pq.write_table(pa.Table.from_pylist(output_rows), path, compression="zstd")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=REPO,
            revision=REVISION,
            allow_patterns=[f"{SUBFOLDER}/*"],
        )
    )
    model_path = snapshot / SUBFOLDER
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()

    validation_manifest = one_file("v6-validation-manifest.parquet")
    validation_audio_path = one_file("v6-validation-audio.parquet")
    validation_rows = pq.read_table(validation_manifest).to_pylist()
    if len(validation_rows) != 6493 or {row["dataset_split"] for row in validation_rows} != {"validation"}:
        raise RuntimeError("unexpected v6 validation manifest")
    validation_audio = pq.read_table(validation_audio_path, columns=["audio"]).column("audio").to_pylist()
    validation_blobs = [validation_audio[int(row["source_row_index"])] for row in validation_rows]
    sinhala_predictions, sinhala_runtime = predict(
        model,
        processor,
        validation_rows,
        validation_blobs,
        language="si",
        batch_size=8,
        num_beams=5,
    )
    sinhala_output = OUTPUT / "yohan-run7-v6-validation-predictions.parquet"
    write_predictions(sinhala_output, validation_rows, sinhala_predictions)
    del validation_audio, validation_blobs

    english_path = one_file("english-retention-test-clean.parquet")
    english_rows = pq.read_table(english_path).to_pylist()
    if len(english_rows) != 2620:
        raise RuntimeError("unexpected English benchmark")
    english_blobs = [row.pop("audio") for row in english_rows]
    english_predictions, english_runtime = predict(
        model,
        processor,
        english_rows,
        english_blobs,
        language="en",
        batch_size=8,
        num_beams=1,
    )
    english_output = OUTPUT / "yohan-run7-english-predictions.parquet"
    write_predictions(english_output, english_rows, english_predictions)

    metadata = {
        "state": "complete",
        "model_repo": REPO,
        "model_revision": REVISION,
        "model_subfolder": SUBFOLDER,
        "v6_validation_rows": len(validation_rows),
        "v6_validation_num_beams": 5,
        "v6_validation_runtime_seconds": sinhala_runtime,
        "v6_validation_predictions_sha256": sha256(sinhala_output),
        "english_rows": len(english_rows),
        "english_num_beams": 1,
        "english_runtime_seconds": english_runtime,
        "english_predictions_sha256": sha256(english_output),
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (OUTPUT / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (WORK / "e016-yohan-run7-result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
