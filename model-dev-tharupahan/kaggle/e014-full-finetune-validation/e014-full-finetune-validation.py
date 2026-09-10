#!/usr/bin/env python3
"""E014: validate E012's winning full-fine-tune config (lr=5e-5, replay=10%)
at 5x E012's step count (500 vs 100, same convention E008->E010 used for the
LoRA rank/LR search), then run the real English-retention check that E012's
own search never measured.

Two stages in one kernel:
1. Train (method="full", lr=5e-5) on the same replay10 manifest E012 used,
   for 500 steps instead of 100 -- same recipe, more steps, not a full
   v5-scale replication (that would need packaging v5's real 199.49h train
   pool for Kaggle, a materially bigger transfer; deferred, documented in
   the E014 report as a real scope decision, not silently assumed identical
   to E010's own full-manifest convention).
2. Score the resulting full-parameter checkpoint against the frozen
   2,620-row LibriSpeech test-clean benchmark -- loaded directly via
   WhisperForConditionalGeneration.from_pretrained(), no PEFT wrapper,
   since this is a full fine-tune checkpoint, not a LoRA adapter (unlike
   every previous English-retention check in this project, which used
   PeftModel.from_pretrained() against an adapter).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
MODEL = WORK / "whisper-small"
TRAIN_OUTPUT = WORK / "e014-train"


def one_file(name: str) -> Path:
    matches = list(INPUTS.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {matches}")
    return matches[0]


def one_dir(name: str) -> Path:
    matches = [p for p in INPUTS.rglob(name) if p.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one dir {name}, found {matches}")
    return matches[0]


def install_runtime(runtime: Path) -> None:
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


def stage_model(runtime: Path) -> None:
    import shutil

    shutil.rmtree(MODEL, ignore_errors=True)
    MODEL.mkdir(parents=True)
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    if not (MODEL / "model.safetensors").is_file():
        raise RuntimeError("offline Whisper-small model is incomplete")


def run_retention_eval(checkpoint_dir: Path, benchmark: Path, output: Path, batch_size: int = 8) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(str(checkpoint_dir), language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        str(checkpoint_dir), torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to("cuda").eval()
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    rows = pq.read_table(benchmark).to_pylist()
    if len(rows) != 2620:
        raise SystemExit(f"unexpected English benchmark size: {len(rows)}")

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
                waveforms, sampling_rate=16000, return_attention_mask=True, return_tensors="pt"
            )
            generated = model.generate(
                inputs.input_features.to("cuda", dtype=torch.float16),
                attention_mask=inputs.attention_mask.to("cuda"),
                max_new_tokens=128,
                no_repeat_ngram_size=3,
            )
            predictions.extend(
                text.strip()
                for text in processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
            print(f"predicted {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    pq.write_table(
        pa.Table.from_pylist(
            [
                {key: value for key, value in row.items() if key != "audio"}
                | {"prediction": prediction, "model": f"{checkpoint_dir}:e014-full-finetune-validation"}
                for row, prediction in zip(rows, predictions)
            ]
        ),
        output,
        compression="zstd",
    )
    return {"rows": len(rows), "runtime_seconds": time.monotonic() - started}


def main() -> None:
    runtime = one_file("whisper-small--model.safetensors").parent
    sinhala_asr_src = one_dir("sinhala_asr")
    train_script = sinhala_asr_src.parent / "train.py"
    if not train_script.is_file():
        raise RuntimeError(f"train.py not found at {train_script}")
    manifest = one_file("manifest-replay10.parquet")
    benchmark = one_file("english-retention-test-clean.parquet")

    install_runtime(runtime)
    stage_model(runtime)

    subprocess_env = os.environ.copy()
    subprocess_env["PYTHONPATH"] = str(sinhala_asr_src.parent)

    # Two arms, per docs/audits/yohan-finetune-lessons.md: Yohan's independent
    # runs show a full fine-tune at lr=3e-5/linear drove English WER 4.3% ->
    # 80.9% (+76.6 pts, severe forgetting), and only lr=1e-5 + cosine kept
    # English intact (+1.79 pts, mild). E012's Sinhala-only search picked
    # lr=5e-5 -- higher than his catastrophic 3e-5. So test both here, each
    # scored against the frozen LibriSpeech benchmark, before trusting either
    # for the full run. Teacher replay (10%, baked into the manifest) is our
    # own addition Yohan does not use -- it may rescue the higher LR, but
    # this measures rather than assumes that.
    arms = [
        {"name": "lr5e-5-linear", "learning_rate": 5e-5, "lr_scheduler_type": "linear"},
        {"name": "lr1e-5-cosine", "learning_rate": 1e-5, "lr_scheduler_type": "cosine"},
    ]

    results = []
    for arm in arms:
        arm_dir = TRAIN_OUTPUT / arm["name"]
        config = {
            "model_name": str(MODEL),
            "manifest": str(manifest),
            "output_dir": str(arm_dir),
            "method": "full",
            "max_steps": 500,
            "learning_rate": arm["learning_rate"],
            "lr_scheduler_type": arm["lr_scheduler_type"],
            "warmup_steps": 50,
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "eval_steps": 500,
            "save_steps": 500,
            "logging_steps": 10,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 2,
            "hourly_price_usd": 0.0,
            "estimated_hours": 0.0,
            "maximum_cost_usd": 0.0,
        }
        arm_dir.mkdir(parents=True, exist_ok=True)
        config_path = arm_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        print(f"=== training arm {arm['name']}: replay=10%, 500 steps ===", flush=True)
        completed = subprocess.run(
            [sys.executable, str(train_script), "--config", str(config_path)],
            env=subprocess_env,
            cwd=str(manifest.parent),
        )
        if completed.returncode != 0:
            raise SystemExit(f"training arm {arm['name']} failed with exit code {completed.returncode}")

        final_dir = arm_dir / "final"
        if not final_dir.is_dir():
            raise RuntimeError(f"expected final checkpoint at {final_dir}")

        print(f"=== arm {arm['name']}: scoring English retention (frozen LibriSpeech benchmark) ===", flush=True)
        retention_output = WORK / f"e014-{arm['name']}-english-retention-predictions.parquet"
        metadata = run_retention_eval(final_dir, benchmark, retention_output)
        metadata["arm"] = arm["name"]
        results.append(metadata)

        # Free disk before the next arm (full-parameter checkpoints are ~3GB;
        # same reason E012's search kernel had to clean up between trials).
        for checkpoint_dir in arm_dir.glob("checkpoint-*"):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(final_dir, ignore_errors=True)

    (WORK / "e014-results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("=== ALL ARMS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
