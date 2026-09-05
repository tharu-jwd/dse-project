"""Generate untouched-Whisper behavior targets for E004 English replay."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
MODEL = WORK / "whisper-small"
OUTPUT = WORK / "e004-english-teacher-labels.parquet"
METADATA = WORK / "e004-english-teacher-labels-metadata.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_parent(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0].parent


def main() -> None:
    runtime = one_parent("whisper-small--model.safetensors")
    source = one_parent("train-manifest.json")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(runtime),
            "transformers==5.16.1",
            "peft==0.20.0",
            "huggingface-hub==1.30.0",
            "tokenizers==0.23.2",
            "safetensors==0.8.0",
            "accelerate==1.14.0",
        ],
        check=True,
    )
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    MODEL.mkdir()
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    manifest_path = source / "train-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for shard in manifest["shards"]:
        path = source / shard["name"]
        if sha256(path) != shard["sha256"]:
            raise RuntimeError(f"training shard hash mismatch: {shard['name']}")
        rows.extend(
            row
            for row in pq.read_table(path).to_pylist()
            if (row.get("decoder_language") or "si") == "en"
        )
    if len(rows) != 1111:
        raise RuntimeError(f"expected 1111 English replay rows, found {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate English replay sample IDs")

    processor = WhisperProcessor.from_pretrained(MODEL, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")
    model.eval()
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    teacher_texts: list[str] = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            waveforms = []
            for row in batch:
                waveform, rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
                if rate != 16000 or waveform.ndim != 1:
                    raise RuntimeError(f"unexpected audio format: {row['sample_id']}")
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
            teacher_texts.extend(
                text.strip()
                for text in processor.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
            )
            print(f"predicted {min(start + 8, len(rows))}/{len(rows)}", flush=True)
    if any(not text for text in teacher_texts):
        raise RuntimeError("empty teacher transcript generated")
    output_rows = [
        {key: value for key, value in row.items() if key != "audio"}
        | {"teacher_text": teacher_text}
        for row, teacher_text in zip(rows, teacher_texts)
    ]
    pq.write_table(pa.Table.from_pylist(output_rows), OUTPUT, compression="zstd")
    metadata = {
        "experiment": "e004",
        "purpose": "untouched Whisper-small behavior-replay targets",
        "source_manifest_sha256": sha256(manifest_path),
        "source_manifest_fingerprint": manifest["source_sha256"],
        "base_model_weights_sha256": sha256(MODEL / "model.safetensors"),
        "rows": len(rows),
        "teacher_labels_sha256": sha256(OUTPUT),
        "runtime_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "decoding": {
            "language": "en",
            "task": "transcribe",
            "batch_size": 8,
            "max_new_tokens": 128,
            "no_repeat_ngram_size": 3,
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
