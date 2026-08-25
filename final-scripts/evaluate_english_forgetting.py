"""Check catastrophic forgetting of English after Sinhala fine-tuning.

Evaluates WER/CER on LibriSpeech test-clean (English) for two models:
  - the base model (default openai/whisper-small)
  - a fine-tuned checkpoint/adapter (this project's Sinhala fine-tune)

Both are forced to language="english", task="transcribe" at generation time
-- the fine-tune scripts (finetune_whisper.py / finetune_whisper_lora.py)
hardcode language="sinhala" into the saved generation_config, so without this
override we'd be asking the fine-tuned model to transcribe English while
told it's Sinhala, a harsher and less informative test than what we actually
want to measure here.

A big WER/CER jump on the fine-tuned model vs. the base model, on the same
English test set, is the forgetting measurement.

Usage (full fine-tune checkpoint):
    python3 evaluate_english_forgetting.py --finetuned /path/to/whisper-small-sinhala/run1

LoRA adapter:
    python3 evaluate_english_forgetting.py \\
        --finetuned /path/to/whisper-small-sinhala-lora/run1 --lora-base openai/whisper-small

Quick pass on a subsample before committing to the full ~2600-row test-clean split:
    python3 evaluate_english_forgetting.py --finetuned <dir> --max-samples 200
"""

import argparse
import io
import json
import time

import evaluate
import jiwer
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TARGET_SR = 16000

NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def load_model(path, processor_repo, is_lora, lora_base, device):
    processor = WhisperProcessor.from_pretrained(processor_repo)
    if is_lora:
        from peft import PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(lora_base)
        model = PeftModel.from_pretrained(base, path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(path)
    # Force English regardless of what the checkpoint's own generation_config says.
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return processor, model.to(device).eval()


def evaluate_model(name, path, processor_repo, is_lora, lora_base, dataset, device, batch_size):
    processor, model = load_model(path, processor_repo, is_lora, lora_base, device)

    predictions, references = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc=name, unit="batch"):
            batch = dataset[i:i + batch_size]
            audios = []
            for row in batch:
                audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if sr != TARGET_SR:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
                audios.append(audio)
            texts = [row["text"] for row in batch]
            inputs = processor.feature_extractor(audios, sampling_rate=TARGET_SR, return_tensors="pt")
            input_features = inputs.input_features.to(device)
            generated_ids = model.generate(input_features, max_new_tokens=225)
            preds = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(preds)
            references.extend(texts)
    elapsed = time.time() - t0

    norm_preds = [NORMALIZE(p) for p in predictions]
    norm_refs = [NORMALIZE(r) for r in references]

    wer = 100 * evaluate.load("wer").compute(predictions=norm_preds, references=norm_refs)
    cer = 100 * evaluate.load("cer").compute(predictions=norm_preds, references=norm_refs)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"wer": wer, "cer": cer, "n_samples": len(references), "elapsed_s": elapsed,
            "predictions": predictions, "references": references}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="openai/whisper-small",
                         help="reference/base model, evaluated for comparison")
    parser.add_argument("--finetuned", required=True,
                         help="path (local dir or HF repo id) to the fine-tuned checkpoint/adapter to check")
    parser.add_argument("--lora-base", default=None,
                         help="if --finetuned is a LoRA adapter, the base model it was trained on "
                              "(e.g. openai/whisper-small); presence of this flag marks --finetuned as LoRA")
    parser.add_argument("--dataset", default="openslr/librispeech_asr",
                         help="HF dataset repo id for the English test set")
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="subsample for a quick pass; omit for the full split (~2620 rows for test-clean)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="english_forgetting_report.txt",
                         help="where to write the human-readable summary report")
    parser.add_argument("--output-json", default=None,
                         help="optional: also write full results (incl. predictions) as JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading English test set: {args.dataset} ({args.dataset_config}/{args.dataset_split})")
    # Streaming: avoids caching the full dataset (all splits/configs) to local disk --
    # we only need a slice of one split. decode=False + manual soundfile decode below
    # sidesteps datasets' torchcodec/ffmpeg audio backend, which isn't installed here.
    ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    col_names = list(next(iter(ds.take(1))).keys())
    if "text" not in col_names:
        # LibriSpeech's HF loaders vary in column naming across mirrors.
        text_col = "sentence" if "sentence" in col_names else "transcription"
        ds = ds.rename_column(text_col, "text")
    if args.max_samples is not None:
        ds = ds.take(args.max_samples)
    ds = list(ds)
    print(f"{len(ds)} rows\n")

    is_lora = args.lora_base is not None

    print(f"=== base: {args.base_model} ===")
    base_r = evaluate_model("base", args.base_model, args.base_model, False, None, ds, device, args.batch_size)
    print(f"  WER: {base_r['wer']:.2f}  CER: {base_r['cer']:.2f}  ({base_r['n_samples']} samples, {base_r['elapsed_s']:.0f}s)\n")

    processor_repo = args.lora_base if is_lora else args.finetuned
    print(f"=== fine-tuned: {args.finetuned} ===")
    ft_r = evaluate_model("finetuned", args.finetuned, processor_repo, is_lora, args.lora_base, ds, device, args.batch_size)
    print(f"  WER: {ft_r['wer']:.2f}  CER: {ft_r['cer']:.2f}  ({ft_r['n_samples']} samples, {ft_r['elapsed_s']:.0f}s)\n")

    wer_delta = ft_r["wer"] - base_r["wer"]
    cer_delta = ft_r["cer"] - base_r["cer"]

    lines = [
        "English catastrophic-forgetting check",
        "======================================",
        f"Dataset: {args.dataset} ({args.dataset_config}/{args.dataset_split}), {len(ds)} samples",
        "",
        f"{'model':<55} {'WER':>8} {'CER':>8}",
        f"{args.base_model:<55} {base_r['wer']:>7.2f}% {base_r['cer']:>7.2f}%",
        f"{args.finetuned:<55} {ft_r['wer']:>7.2f}% {ft_r['cer']:>7.2f}%",
        "",
        f"WER delta (finetuned - base): {wer_delta:+.2f} pts",
        f"CER delta (finetuned - base): {cer_delta:+.2f} pts",
    ]
    report = "\n".join(lines)
    print(report)

    with open(args.output, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote report -> {args.output}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"base": base_r, "finetuned": ft_r, "wer_delta": wer_delta, "cer_delta": cer_delta}, f, indent=2)
        print(f"wrote full results -> {args.output_json}")


if __name__ == "__main__":
    main()
