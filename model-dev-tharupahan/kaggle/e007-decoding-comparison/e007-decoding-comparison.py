"""Compare greedy vs. beam-search decoding on E007's real final adapter,
using the exact same evaluation code path (batched, fp16, attention_mask,
max_new_tokens=64) as E007's own original Sinhala validation scoring in
run_e002_colab.py's generate_validation() -- so the numbers are directly
comparable to E007's own reported 81.71% canonical WER, unlike an earlier
attempt on different hardware (Camber) that reproduced a materially
different greedy baseline for reasons never fully isolated (see
docs/project/plan.md item 1 and docs/audits for the trace)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "e007-decoding-comparison"
MODEL = WORK / "whisper-small"
ADAPTER_SHA256 = "4c92dee2fbdc7bc59ef70e7e17654967b47cf8b535956f67a09c2c1458acc865"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_file(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0]


def generate_at(
    model: Any,
    processor: Any,
    rows: list[dict],
    num_beams: int,
    batch_size: int = 8,
) -> list[str]:
    predictions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            waveforms = []
            for row in batch:
                waveform, rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
                if rate != 16000 or waveform.ndim != 1:
                    raise ValueError(f"unexpected validation audio: {row['sample_id']}")
                waveforms.append(waveform)
            inputs = processor.feature_extractor(
                waveforms, sampling_rate=16000, return_attention_mask=True, return_tensors="pt"
            )
            generated = model.generate(
                inputs.input_features.to("cuda", dtype=torch.float16),
                attention_mask=inputs.attention_mask.to("cuda"),
                max_new_tokens=64,
                no_repeat_ngram_size=3,
                num_beams=num_beams,
            )
            predictions.extend(
                text.strip()
                for text in processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
    return predictions


def main() -> None:
    runtime = one_file("whisper-small--model.safetensors").parent
    validation_bundle = one_file("v4-validation-206-audio.parquet")
    result_path = one_file("e007-phase-b-kaggle-result.json")
    training_result = json.loads(result_path.read_text())
    if training_result["adapter_sha256"] != ADAPTER_SHA256:
        raise RuntimeError("E007 phase-b training result adapter hash mismatch")
    adapter = result_path.parent / "e007-phase-b/final-adapter"
    if sha256(adapter / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise RuntimeError("attached E007 adapter hash mismatch")

    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
            "--find-links", str(runtime),
            "transformers==5.16.1", "peft==0.20.0", "huggingface-hub==1.30.0",
            "tokenizers==0.23.2", "safetensors==0.8.0", "accelerate==1.14.0",
        ],
        check=True,
    )

    MODEL.mkdir()
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)

    processor = WhisperProcessor.from_pretrained(str(MODEL), language="si", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(
        str(MODEL), torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base.generation_config.language = "si"
    base.generation_config.task = "transcribe"
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.to("cuda")
    model.config.use_cache = True
    model.eval()

    validation = pq.read_table(validation_bundle).to_pylist()
    if len(validation) != 206:
        raise ValueError("validation row count mismatch")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for num_beams in (1, 5):
        predictions = generate_at(model, processor, validation, num_beams)
        rows = [
            {key: value for key, value in row.items() if key != "audio"} | {"prediction": prediction}
            for row, prediction in zip(validation, predictions, strict=True)
        ]
        table = pa.Table.from_pylist(rows)
        out_path = OUTPUT / f"predictions-beam{num_beams}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        results[str(num_beams)] = {
            "predictions_sha256": sha256(out_path),
            "sample_predictions": [
                {"reference": r["reference"], "prediction": r["prediction"]} for r in rows[:5]
            ],
        }
        print(f"num_beams={num_beams} done, wrote {out_path}")

    (OUTPUT / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
