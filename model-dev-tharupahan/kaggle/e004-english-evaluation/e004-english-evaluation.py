"""Evaluate E004 on the frozen LibriSpeech test-clean benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "e004-english-evaluation"
MODEL = WORK / "whisper-small"
ADAPTER_SHA256 = "29f01b70a7a62abd6050c688bf1a09a8979429dd95eb0fe098237fce7803ae1c"


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


def main() -> None:
    runtime = one_file("whisper-small--model.safetensors").parent
    benchmark = one_file("english-retention-test-clean.parquet")
    evaluator = one_file("run_english_retention.py")
    result_path = one_file("e004-kaggle-training-result.json")
    training_result = json.loads(result_path.read_text())
    if training_result["adapter_sha256"] != ADAPTER_SHA256:
        raise RuntimeError("E004 training result adapter hash mismatch")
    adapter = result_path.parent / "e004-training/final-adapter"
    if sha256(adapter / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise RuntimeError("attached E004 adapter hash mismatch")
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
    MODEL.mkdir()
    for asset in runtime.glob("whisper-small--*"):
        (MODEL / asset.name.removeprefix("whisper-small--")).symlink_to(asset)
    environment = os.environ.copy()
    environment["SINHALA_ASR_MODEL"] = str(MODEL)
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--benchmark",
            str(benchmark),
            "--output-dir",
            str(OUTPUT),
            "--adapter",
            str(adapter),
            "--variant",
            # The frozen evaluator snapshot predates the E004 display label.
            # Run its identical adapter path under the supported E003 label,
            # then correct only provenance fields and filenames below.
            "e003",
        ],
        check=True,
        env=environment,
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    old_predictions = OUTPUT / "e003-predictions.parquet"
    predictions = OUTPUT / "e004-predictions.parquet"
    table = pq.read_table(old_predictions)
    model_index = table.schema.get_field_index("model")
    table = table.set_column(
        model_index,
        "model",
        pa.array([f"{MODEL}:e004"] * table.num_rows),
    )
    pq.write_table(table, predictions, compression="zstd")
    old_predictions.unlink()
    old_runtime = OUTPUT / "e003-runtime.json"
    runtime_metadata = json.loads(old_runtime.read_text())
    runtime_metadata.update(
        {
            "variant": "e004",
            "predictions_sha256": sha256(predictions),
        }
    )
    (OUTPUT / "e004-runtime.json").write_text(
        json.dumps(runtime_metadata, indent=2) + "\n", encoding="utf-8"
    )
    old_runtime.unlink()
    result = {
        "state": "complete",
        "rows": runtime_metadata["rows"],
        "adapter_sha256": ADAPTER_SHA256,
        "predictions_sha256": runtime_metadata["predictions_sha256"],
        "benchmark_sha256": runtime_metadata["benchmark_sha256"],
    }
    (WORK / "e004-kaggle-english-evaluation-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
