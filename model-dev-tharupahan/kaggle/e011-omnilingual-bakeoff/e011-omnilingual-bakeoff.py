"""E011: zero-training model-family bake-off, Omnilingual ASR CTC vs. E007.

Item 3 in docs/project/plan.md's ranked execution plan. Runs Meta's
Omnilingual ASR CTC checkpoints (Apache 2.0, no fine-tuning) on the exact
same 206-row v4 validation audio/references E007 was scored against, so
the result is directly comparable via this project's own metrics without
retraining anything. E007's own numbers are not recomputed here -- they
are already hash-verified in docs/experiments/e007-...md and were scored
with the same validation parquet and the same metrics module; this kernel
only needs to produce Omnilingual's predictions on that same audio.

Deliberate deviation from this project's usual enable_internet=false
convention: `pip install omnilingual-asr` and the model checkpoint
download both need internet access, and there is no offline wheelhouse
practical to pre-stage for fairseq2's platform/CUDA-specific build (unlike
the pure-PyTorch runtime bundle E003 onward has pre-staged). Documented in
docs/training/kaggle-cli.md rather than left silent.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INPUTS = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "e011-omnilingual-bakeoff"

# Both fit comfortably in a T4's 16 GiB even loaded one after another
# (~2 GiB and ~3 GiB respectively per the project's own upstream README
# check); LLM-ASR 1B/7B are skipped for this first pass -- much larger
# downloads and slower decode, not justified unless CTC is competitive.
MODEL_CARDS = ["omniASR_CTC_300M_v2", "omniASR_CTC_1B_v2"]


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
    validation_bundle = one_file("v4-validation-206-audio.parquet")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "omnilingual-asr"],
        check=True,
    )
    # Real, first-run failure (kernel version 1): omnilingual-asr's own
    # dependency resolution silently downgrades numpy to 1.26.4, which is
    # ABI-incompatible with Kaggle's preinstalled torch/etc (compiled
    # against numpy 2.x) -- "numpy.dtype size changed, may indicate binary
    # incompatibility" the moment anything touches torch._dynamo. Force
    # numpy back to the 2.x line Kaggle's own stack expects; omnilingual's
    # numpy<2 pin looks defensive rather than load-bearing (nothing in its
    # own import chain fails at the numpy-2-vs-1 level, only downstream
    # torch internals that break for lack of numpy 2.x).
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", "numpy>=2,<3"],
        check=True,
    )
    # Import after install -- the package does not exist in the base image.
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    validation = pq.read_table(validation_bundle).to_pylist()
    if len(validation) != 206:
        raise ValueError("validation row count mismatch")
    audio_bytes = [row["audio"] for row in validation]
    langs = ["sin_Sinh"] * len(validation)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_card in MODEL_CARDS:
        print(f"loading {model_card}", flush=True)
        # float32, not the library default bf16 -- T4 (Turing) has no
        # native bf16 tensor-core support; avoid relying on emulation for
        # a number we intend to trust.
        pipeline = ASRInferencePipeline(model_card=model_card, dtype=torch.float32)
        predictions = pipeline.transcribe(audio_bytes, lang=langs, batch_size=8)
        rows = [
            {
                "sample_id": row["sample_id"],
                "speaker_id": row.get("speaker_id"),
                "language_class": row.get("language_class"),
                "duration_seconds": row.get("duration_seconds"),
                "reference": row["reference"],
                "audio_sha256": hashlib.sha256(row["audio"]).hexdigest(),
                "prediction": prediction.strip(),
                "model": model_card,
            }
            for row, prediction in zip(validation, predictions, strict=True)
        ]
        table = pa.Table.from_pylist(rows)
        out_path = OUTPUT / f"predictions-{model_card}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        results[model_card] = {
            "predictions_sha256": sha256(out_path),
            "sample_predictions": [
                {"reference": r["reference"], "prediction": r["prediction"]} for r in rows[:5]
            ],
        }
        print(f"{model_card} done, wrote {out_path}", flush=True)
        del pipeline
        torch.cuda.empty_cache()

    (OUTPUT / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
