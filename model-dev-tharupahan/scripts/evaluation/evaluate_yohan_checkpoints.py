#!/usr/bin/env python3
"""Independently evaluate Yohan2003/whisper-small-sinhala checkpoints.

Two evaluation sets, run separately, both against this project's own metrics
(sinhala_asr.evaluation.metrics.score_pair/aggregate, strict_normalize):

1. This project's own frozen v4 validation split (206 rows, speaker-disjoint,
   OpenSLR-52) -- the same benchmark every E000-E009 result is measured
   against, for direct comparability.
2. A sample of the Path Nirvana Sinhala TTS dataset (metadata.csv + wavs/,
   downloaded separately from github.com/pnfo/sinhala-tts-dataset releases,
   v2.1) -- genuinely unseen audio for these checkpoints (they were trained
   on OpenSLR-52/YouTube/BizBrains/Linga per their own model card, not this).
   Not a like-for-like WER comparison to (1) -- different domain (read TTS
   prompts including Pali/Sanskrit-origin vocabulary, studio-quality 22050Hz
   audio, not conversational OpenSLR-52 speech) -- but a real generalization
   check the OpenSLR-only number cannot give.

Every generate() call sets no_repeat_ngram_size=3 (see
docs/audits/e008-eval-repetition-bug.md for why: undertrained/unstable
checkpoints can otherwise degenerate into a runaway repeated-token loop that
inflates WER/CER past what the model's real quality would produce). Also
prints a handful of raw predictions per run so the aggregate number is never
trusted blind.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from sinhala_asr.evaluation.metrics import aggregate, score_pair, strict_normalize
from sinhala_asr.training.dataset import ManifestAudioDataset, load_training_rows

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_model(model_path: Path, adapter: bool) -> tuple[Any, WhisperProcessor]:
    processor = WhisperProcessor.from_pretrained(str(model_path), language="si", task="transcribe")
    if adapter:
        base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        model = PeftModel.from_pretrained(base, str(model_path)).merge_and_unload()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(str(model_path))
    model.to(DEVICE).eval()
    model.generation_config.language = "si"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.no_repeat_ngram_size = 3
    return model, processor


def transcribe(model: Any, processor: WhisperProcessor, audio: Any) -> str:
    feats = processor.feature_extractor([audio], sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(feats.input_features.to(DEVICE), max_new_tokens=225)
    return processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def load_v4_validation() -> list[dict[str, Any]]:
    _, validation = load_training_rows(Path("data/versions/v4/manifest.parquet"))
    dataset = ManifestAudioDataset(validation)
    rows = []
    for i in range(len(dataset)):
        item = dataset[i]
        rows.append({"audio": item["audio"], "reference": item["text"], "id": item["sample_id"]})
    return rows


def load_tts_sample(tts_dir: Path, sample_size: int, seed: int) -> list[dict[str, Any]]:
    with (tts_dir / "metadata.csv").open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="|")
        rows = [row for row in reader if len(row) >= 3]
    rng = random.Random(seed)
    sample = rng.sample(rows, min(sample_size, len(rows)))
    result = []
    for row in sample:
        row_id, _romanized, sinhala_text, *rest = row
        wav_path = tts_dir / "wavs" / f"{row_id}.wav"
        if not wav_path.is_file():
            continue
        audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != 16000:
            from scipy.signal import resample

            audio = resample(audio, int(len(audio) * 16000 / sample_rate)).astype("float32")
        result.append(
            {
                "audio": audio,
                "reference": sinhala_text,
                "id": row_id,
                "speaker": rest[0] if rest else "unknown",
            }
        )
    return result


def run_eval(model: Any, processor: WhisperProcessor, rows: list[dict[str, Any]], label: str) -> None:
    scored = []
    print(f"\n=== {label}: {len(rows)} rows ===")
    for i, row in enumerate(rows):
        prediction = transcribe(model, processor, row["audio"])
        score = score_pair(row["reference"], prediction, strict_normalize)
        scored.append({f"strict_{key}": value for key, value in score.items()})
        if i < 5:
            print(f"  REF:  {row['reference']}")
            print(f"  HYP:  {prediction}")
            print()
    agg = aggregate(scored, "strict")
    print(f"{label}: wer={agg['wer'] * 100:.2f}% cer={agg['cer'] * 100:.2f}% (n={agg['rows']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--tts-dir", type=Path, default=Path(
        "/Users/tharupahan/Code/ASR-prior-works/sinhala-tts-dataset/v2.1-audio"
    ))
    parser.add_argument("--tts-sample-size", type=int, default=300)
    parser.add_argument("--tts-seed", type=int, default=20260906)
    parser.add_argument("--skip-v4", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()

    print(f"loading {args.label} from {args.model} (adapter={args.adapter}) on {DEVICE}")
    model, processor = load_model(args.model, args.adapter)

    if not args.skip_v4:
        v4_rows = load_v4_validation()
        run_eval(model, processor, v4_rows, f"{args.label} / v4-validation")

    if not args.skip_tts:
        tts_rows = load_tts_sample(args.tts_dir, args.tts_sample_size, args.tts_seed)
        run_eval(model, processor, tts_rows, f"{args.label} / tts-sample")


if __name__ == "__main__":
    main()
