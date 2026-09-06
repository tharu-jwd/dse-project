#!/usr/bin/env python3
"""Compare greedy vs. beam-search decoding on an already-trained adapter.

Zero training cost -- re-runs inference on an existing checkpoint with
different `num_beams` values and scores each with this project's own
metrics. Item 1 in docs/project/plan.md's forward-looking priority order:
ranked first specifically because it composes with every other lever
(whichever checkpoint eventually wins, this choice applies on top of it)
and because no evaluation in this project through E010 has used anything
but greedy decoding.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from sinhala_asr.evaluation.metrics import aggregate, metric_normalize, score_pair
from sinhala_asr.training.dataset import ManifestAudioDataset, load_training_rows

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_model(adapter_dir: Path) -> tuple[Any, WhisperProcessor]:
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="si", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    model.to(DEVICE).eval()
    model.generation_config.language = "si"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.no_repeat_ngram_size = 3
    return model, processor


def run(model: Any, processor: WhisperProcessor, rows: list[dict[str, Any]], num_beams: int) -> dict[str, Any]:
    started = time.monotonic()
    predictions = []
    scored = []
    for row in rows:
        feats = processor.feature_extractor([row["audio"]], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            out = model.generate(feats.input_features.to(DEVICE), max_length=225, num_beams=num_beams)
        text = processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        predictions.append({"sample_id": row["sample_id"], "reference": row["text"], "prediction": text})
        score = score_pair(row["text"], text, metric_normalize)
        scored.append({f"canonical_{k}": v for k, v in score.items()})
    agg = aggregate(scored, "canonical")
    runtime = time.monotonic() - started
    return {
        "num_beams": num_beams,
        "wer": agg["wer"],
        "cer": agg["cer"],
        "rows": agg["rows"],
        "runtime_seconds": runtime,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/versions/v4/manifest.parquet"))
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"loading adapter from {args.adapter} on {DEVICE}", flush=True)
    model, processor = load_model(args.adapter)

    _, validation_rows = load_training_rows(args.manifest)
    dataset = ManifestAudioDataset(validation_rows)
    rows = [dataset[i] for i in range(len(dataset))]
    print(f"validation rows: {len(rows)}", flush=True)

    results = []
    for num_beams in args.beam_sizes:
        result = run(model, processor, rows, num_beams)
        print(
            f"num_beams={num_beams}: canonical WER={result['wer']*100:.2f}% "
            f"CER={result['cer']*100:.2f}% ({result['runtime_seconds']:.1f}s, "
            f"{result['runtime_seconds']/result['rows']:.2f}s/clip)",
            flush=True,
        )
        print("first 5 predictions:", flush=True)
        for p in result["predictions"][:5]:
            print(f"  REF: {p['reference']}", flush=True)
            print(f"  HYP: {p['prediction']}", flush=True)
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
