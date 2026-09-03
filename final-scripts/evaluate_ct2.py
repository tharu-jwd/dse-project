"""Score a CTranslate2 (faster-whisper) export -- e.g. an int8-quantized
Sinhala checkpoint -- against the same test set and normalization pipeline
used by evaluate_finetuned.py, so WER/CER numbers are directly comparable.

evaluate_finetuned.py can't load this model shape: it only knows HF
`from_pretrained` (full fine-tune) and PEFT (LoRA) checkpoints, not a
CTranslate2 model.bin/vocabulary.json export. This script fills that gap.

Usage:
    python3 evaluate_ct2.py --model ../models/whisper-sinhala1-ct2 --device cuda --compute-type int8
"""

import argparse
import csv
import io
import os
import time

import evaluate
import jiwer
import pyarrow.parquet as pq
import soundfile as sf
from faster_whisper import WhisperModel
from tqdm import tqdm

TARGET_SR = 16000
TEST_PARQUET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stratified", "test.parquet")

NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def _read_parquet_table(path, columns):
    is_remote = path.startswith("gs://")
    return pq.read_table(path, columns=columns, memory_map=not is_remote)


def count_rows(parquet_path, max_samples=None):
    n = pq.read_metadata(parquet_path).num_rows
    return n if max_samples is None else min(max_samples, n)


def iter_rows(parquet_path, max_samples=None):
    """Streams row groups and yields one decoded sample at a time -- the full
    audio column doesn't fit in RAM on a small machine, so never materialize
    it as a list."""
    is_remote = parquet_path.startswith("gs://")
    pf = pq.ParquetFile(parquet_path)
    yielded = 0
    for batch in pf.iter_batches(columns=["audio", "text"], batch_size=8):
        for audio_bytes, text in zip(batch.column("audio").to_pylist(), batch.column("text").to_pylist()):
            if max_samples is not None and yielded >= max_samples:
                return
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if sr != TARGET_SR:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            yield audio, text
            yielded += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="CTranslate2 model dir (model.bin + vocabulary.json)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="int8", help="e.g. int8, int8_float16, float16, float32")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    print(f"Loading test set: {TEST_PARQUET}")
    n_rows = count_rows(TEST_PARQUET, max_samples=args.max_samples)
    print(f"{n_rows} rows\n")

    print(f"Loading model {args.model} (device={args.device}, compute_type={args.compute_type})")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    predictions, references = [], []
    t0 = time.time()
    for audio, text in tqdm(iter_rows(TEST_PARQUET, max_samples=args.max_samples), total=n_rows, desc=args.model, unit="sample"):
        segments, _ = model.transcribe(audio, language="si", task="transcribe")
        pred = "".join(seg.text for seg in segments)
        predictions.append(pred)
        references.append(text)
    elapsed = time.time() - t0

    norm_preds = [NORMALIZE(p) for p in predictions]
    norm_refs = [NORMALIZE(r) for r in references]

    wer = 100 * evaluate.load("wer").compute(predictions=norm_preds, references=norm_refs)
    cer = 100 * evaluate.load("cer").compute(predictions=norm_preds, references=norm_refs)

    print(f"\n=== {args.model} ({args.compute_type}) ===")
    print(f"  WER: {wer:.2f}  CER: {cer:.2f}  ({len(references)} samples, {elapsed:.0f}s)")

    os.makedirs(args.output_dir, exist_ok=True)
    safe_name = args.model.strip("/").replace("/", "_")
    out_path = os.path.join(args.output_dir, f"{safe_name}_{args.compute_type}_predictions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reference", "prediction"])
        for ref, pred in zip(references, predictions):
            writer.writerow([ref, pred])
    print(f"wrote per-sample predictions -> {out_path}")


if __name__ == "__main__":
    main()
