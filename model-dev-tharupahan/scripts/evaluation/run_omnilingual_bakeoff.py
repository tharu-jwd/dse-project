#!/usr/bin/env python3
"""E011 zero-training bake-off: Omnilingual ASR CTC vs. E007, on Camber.

Moved here from kaggle/e011-omnilingual-bakeoff/ after two real,
documented Kaggle failures (see docs/training/kaggle-cli.md): Kaggle's
heavily pre-loaded image kept fighting omnilingual-asr's own dependency
resolution (numpy ABI mismatch, then a torchaudio/CUDA-runtime mismatch).
Camber's base engine ships almost nothing (docs/training/camber-cli.md),
so `pip install omnilingual-asr` resolves a single consistent dependency
set from scratch instead of colliding with a pre-existing, differently
pinned stack -- a better fit for this one library, not a general platform
preference change.

Uses this project's own manifest loader directly (same v4 manifest and
raw audio already staged on the Camber stash for E008/E009), not a
separate exported bundle -- so "identical audio, references,
normalization" against E007 holds by construction: E007 was scored from
the same manifest's validation split via the same normalization module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sinhala_asr.training.dataset import ManifestAudioDataset, load_training_rows

# Both fit comfortably even loaded one after another on a single GPU
# (~2 GiB and ~3 GiB respectively per the upstream project's own README).
# LLM-ASR 1B/7B are skipped for this first pass -- much larger downloads
# and slower decode, not justified unless CTC is already competitive.
MODEL_CARDS = ["omniASR_CTC_300M_v2", "omniASR_CTC_1B_v2"]


def _extend_ld_library_path_for_torch_cuda_libs() -> None:
    """Work around torchaudio's extension not finding torch's own CUDA libs.

    Real, reproduced failure (two Camber attempts, and both Kaggle
    attempts before this script moved off Kaggle): `pip install
    omnilingual-asr` resolves `torch` and `torchaudio` independently
    (neither is version-pinned by omnilingual-asr or fairseq2), which can
    pick a torchaudio release expecting a different CUDA runtime than
    what ends up on LD_LIBRARY_PATH -- `OSError: libcudart.so.13: cannot
    open shared object file`, thrown from `torchaudio`'s own compiled
    extension via a bare `ctypes.CDLL` inside `torch.ops.load_library`.
    torch's own internal CUDA ops don't hit this (torch's own .so files
    carry an RPATH to their own bundled `nvidia-*-cuXX` pip packages),
    but a separate extension calling `ctypes.CDLL` directly does not
    inherit that RPATH -- it needs the directory on `LD_LIBRARY_PATH`
    instead. `ctypes.CDLL`/`dlopen` re-consult the *current* environment
    on each call, so setting this here, before importing anything that
    triggers the load, is enough -- no need for the shell that launched
    Python to have set it first.
    """
    import glob
    import importlib
    import os

    import torch

    directories = [os.path.join(os.path.dirname(torch.__file__), "lib")]
    for nvidia_dir in glob.glob(
        os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "*", "lib")
    ):
        directories.append(os.path.abspath(nvidia_dir))
    # Also cover the case where the nvidia-* packages are siblings of
    # torch rather than nested under it (varies by pip/platform layout).
    try:
        import nvidia

        for nvidia_dir in glob.glob(os.path.join(os.path.dirname(nvidia.__file__), "*", "lib")):
            directories.append(os.path.abspath(nvidia_dir))
    except ImportError:
        pass
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(directories + ([existing] if existing else []))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/versions/v4/manifest.parquet"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-cards", nargs="+", default=MODEL_CARDS)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch

    _extend_ld_library_path_for_torch_cuda_libs()
    import os

    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}", flush=True)
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    _, validation_rows = load_training_rows(args.manifest)
    if len(validation_rows) != 206:
        raise ValueError(f"expected 206 validation rows, found {len(validation_rows)}")
    dataset = ManifestAudioDataset(validation_rows)
    items = [dataset[i] for i in range(len(dataset))]
    # Plain 1-D (time,) -- confirmed from the pipeline's own source
    # (resample_to_16khz's channel/time transpose logic only runs when
    # current_sample_rate != target_sample_rate, which is never true here
    # since this project's audio is already 16 kHz, so that logic never
    # even executes). A prior attempt's defensive (1, N) channels-first
    # reshape was wrong and caused a real, confirmed failure downstream:
    # the batch collation treated the leading size-1 axis as something
    # other than channels, producing a Conv1d input of length 1 per item
    # ("Calculated padded input size per channel: (1). Kernel size: (10)").
    audio_inputs = [
        {"waveform": item["audio"], "sample_rate": 16000} for item in items
    ]
    langs = ["sin_Sinh"] * len(items)
    print(f"loaded {len(items)} validation rows", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_card in args.model_cards:
        print(f"loading {model_card}", flush=True)
        pipeline = ASRInferencePipeline(model_card=model_card, dtype=torch.float32)
        predictions = pipeline.transcribe(audio_inputs, lang=langs, batch_size=args.batch_size)
        rows = [
            {
                "sample_id": item["sample_id"],
                "speaker_id": manifest_row.get("speaker_id"),
                "language_class": manifest_row.get("language_class"),
                "duration_seconds": manifest_row.get("duration_seconds"),
                "reference": item["text"],
                "audio_sha256": sha256_bytes(item["audio"].tobytes()),
                "prediction": prediction.strip(),
                "model": model_card,
            }
            for item, manifest_row, prediction in zip(items, validation_rows, predictions, strict=True)
        ]
        table = pa.Table.from_pylist(rows)
        out_path = args.output_dir / f"predictions-{model_card}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        results[model_card] = {
            "predictions_sha256": sha256_file(out_path),
            "sample_predictions": [
                {"reference": r["reference"], "prediction": r["prediction"]} for r in rows[:5]
            ],
        }
        print(f"{model_card} done, wrote {out_path}", flush=True)
        del pipeline
        torch.cuda.empty_cache()

    (args.output_dir / "result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
