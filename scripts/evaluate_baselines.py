"""Benchmark WER/CER of published Sinhala Whisper models against our own
`stratified/test.parquet`, before fine-tuning our own model -- so the numbers
in `finetune_whisper.py`'s eventual output are comparable against a fixed,
known reference point rather than whatever each model's own paper/card
reports (different eval sets, different text normalization -- not
apples-to-apples otherwise).

Default models evaluated (see MODEL_SPECS below):
  - openai/whisper-small           -- unfinetuned base, per the proposal's
                                       "original multilingual Whisper" baseline
  - Lingalingeswaran/whisper-small-sinhala -- fully fine-tuned whisper-small,
                                       the proposal's "existing published
                                       Sinhala fine-tune" baseline
  - SPEAK-ASR/whisper-si-exp-10    -- LoRA adapter on openai/whisper-small,
                                       SPEAK-ASR's most recent small-based
                                       iteration (card reports WER 12.70 on
                                       their own eval set)

SPEAK-ASR also publishes whisper-*medium*-based models
(whisper-si-exp-10-medium[-all], whisper-medium-si-merged) -- a different
size class than what we're fine-tuning, so not included by default, but
addable via --models (see --list-models).

Model loading handles two shapes: "full" (directly `from_pretrained`-able)
and "lora" (a PEFT adapter that must be applied on top of its base model).

Text normalization before scoring (via jiwer's transform pipeline: lowercase,
strip punctuation, collapse whitespace) -- otherwise WER differences driven
by cosmetic punctuation/casing choices would swamp real transcription
differences between models.

Usage:
    python3 dse-project/scripts/evaluate_baselines.py
    python3 dse-project/scripts/evaluate_baselines.py --max-samples 500 --batch-size 16
    python3 dse-project/scripts/evaluate_baselines.py --list-models
    python3 dse-project/scripts/evaluate_baselines.py --models openai/whisper-small SPEAK-ASR/whisper-si-exp-9
"""

import argparse
import io
import time

import evaluate
import jiwer
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_TEST_PARQUET = "dse-project/model-development/data/final_split_dataset/stratified/test.parquet"
TARGET_SR = 16000

MODEL_SPECS = {
    "openai/whisper-small": {"type": "full", "processor_repo": "openai/whisper-small"},
    "Lingalingeswaran/whisper-small-sinhala": {"type": "full", "processor_repo": "Lingalingeswaran/whisper-small-sinhala"},
    "SPEAK-ASR/whisper-si-exp-10": {"type": "lora", "base_model": "openai/whisper-small", "processor_repo": "openai/whisper-small"},
    # different size class, not fine-tuned to compare 1:1 against whisper-small, but available:
    "SPEAK-ASR/whisper-si-exp-10-medium-all": {"type": "lora", "base_model": "openai/whisper-medium", "processor_repo": "openai/whisper-medium"},
    "SPEAK-ASR/whisper-medium-si-merged": {"type": "full", "processor_repo": "SPEAK-ASR/whisper-medium-si-merged"},
}
DEFAULT_MODELS = [
    "openai/whisper-small",
    "Lingalingeswaran/whisper-small-sinhala",
    "SPEAK-ASR/whisper-si-exp-10",
]

NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


class EvalAudioDataset(torch.utils.data.Dataset):
    """Minimal (audio_array, reference_text) reader -- no tokenization here,
    since each model under test brings its own processor/tokenizer."""

    def __init__(self, parquet_path, max_samples=None):
        self.table = pq.read_table(parquet_path, columns=["audio", "text"], memory_map=True)
        self.n = self.table.num_rows if max_samples is None else min(max_samples, self.table.num_rows)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        audio_bytes = self.table.column("audio")[idx].as_py()
        text = self.table.column("text")[idx].as_py()
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        return audio, text


def collate_eval(batch):
    audios, texts = zip(*batch)
    return list(audios), list(texts)


def load_model(repo_id, spec, device):
    processor = WhisperProcessor.from_pretrained(spec["processor_repo"])
    if spec["type"] == "full":
        model = WhisperForConditionalGeneration.from_pretrained(repo_id)
    elif spec["type"] == "lora":
        from peft import PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(spec["base_model"])
        model = PeftModel.from_pretrained(base, repo_id)
    else:
        raise ValueError(f"unknown model type: {spec['type']}")
    model.generation_config.language = "sinhala"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return processor, model.to(device).eval()


def evaluate_model(repo_id, spec, dataset, device, batch_size):
    processor, model = load_model(repo_id, spec, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_eval)

    predictions, references = [], []
    t0 = time.time()
    with torch.no_grad():
        for audios, texts in loader:
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

    return {
        "wer": wer,
        "cer": cer,
        "n_samples": len(references),
        "elapsed_s": elapsed,
        "predictions": predictions,
        "references": references,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-parquet", default=DEFAULT_TEST_PARQUET)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                         help="repo IDs to evaluate; must be present in MODEL_SPECS (see --list-models)")
    parser.add_argument("--max-samples", type=int, default=None, help="subsample for a quick pass; omit for the full test set")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-csv", default="baseline_predictions.csv",
                         help="per-sample predictions/references for the last-evaluated model in each run, for error inspection")
    parser.add_argument("--list-models", action="store_true", help="print known model specs and exit")
    parser.add_argument("--custom-model", action="append", default=[],
                         help="local checkpoint dir (e.g. your own fine-tuned --output-dir) to evaluate as a "
                              "full model, on top of --models; repeatable")
    args = parser.parse_args()

    for path in args.custom_model:
        MODEL_SPECS[path] = {"type": "full", "processor_repo": path}
        if path not in args.models:
            args.models = args.models + [path]

    if args.list_models:
        for repo_id, spec in MODEL_SPECS.items():
            print(f"{repo_id}  ({spec['type']}" + (f", base={spec['base_model']}" if spec['type'] == 'lora' else "") + ")")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading test set: {args.test_parquet}")
    dataset = EvalAudioDataset(args.test_parquet, max_samples=args.max_samples)
    print(f"{len(dataset)} rows\n")

    results = {}
    for repo_id in args.models:
        if repo_id not in MODEL_SPECS:
            print(f"skipping {repo_id}: not in MODEL_SPECS, add it or use a known repo_id (see --list-models)")
            continue
        print(f"=== {repo_id} ===")
        spec = MODEL_SPECS[repo_id]
        r = evaluate_model(repo_id, spec, dataset, device, args.batch_size)
        results[repo_id] = r
        print(f"  WER: {r['wer']:.2f}  CER: {r['cer']:.2f}  "
              f"({r['n_samples']} samples, {r['elapsed_s']:.0f}s)\n")

    print("=== Summary ===")
    print(f"{'model':<45} {'WER':>8} {'CER':>8}")
    for repo_id, r in results.items():
        print(f"{repo_id:<45} {r['wer']:>7.2f}% {r['cer']:>7.2f}%")

    if results:
        import csv
        last_repo, last_r = list(results.items())[-1]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "reference", "prediction"])
            for ref, pred in zip(last_r["references"], last_r["predictions"]):
                writer.writerow([last_repo, ref, pred])
        print(f"\nwrote per-sample predictions for {last_repo} -> {args.output_csv}")


if __name__ == "__main__":
    main()
