"""Score your own fine-tuned Whisper checkpoints/adapters against the final
Sinhala test set, to pick the best one out of however many
`finetune_whisper.py` / `finetune_whisper_lora.py` runs you've done.

Model loading handles two shapes:
  - "full"  -- a `finetune_whisper.py` --output-dir, directly `from_pretrained`-able
  - "lora"  -- a `finetune_whisper_lora.py` --output-dir, a PEFT adapter that
               must be applied on top of its base model (e.g. openai/whisper-small)

Pass as many of each as you want to compare in one run -- the summary at the
end is sorted by WER (best first), so the top row is your answer.

Text normalization before scoring (via jiwer's transform pipeline: lowercase,
strip punctuation, collapse whitespace) -- otherwise WER differences driven
by cosmetic punctuation/casing choices would swamp real transcription
differences between runs.

The test-set path is NOT a CLI flag -- same as the fine-tune scripts, it
expects `model-development/data/stratified/test.parquet` to exist (see
`model-development/README.md` for the expected layout).

Usage: compare a full fine-tune against a LoRA run
    python3 evaluation/evaluate_finetuned.py \\
        --model /workspace/whisper-small-sinhala/run6-lr3e-5-bs32 \\
        --lora /workspace/whisper-small-sinhala-lora/run1-lr1e-4-bs32:openai/whisper-small

Compare several checkpoints of the same kind (e.g. different epochs/LRs from
a hyperparameter sweep) to find the best one:
    python3 evaluation/evaluate_finetuned.py \\
        --model /workspace/whisper-small-sinhala/run1-lr1e-5-bs16 \\
        --model /workspace/whisper-small-sinhala/run2-lr3e-5-bs16 \\
        --model /workspace/whisper-small-sinhala/run3-lr3e-5-bs32

Quick sanity pass on a subsample before committing to a full test-set run:
    python3 evaluation/evaluate_finetuned.py --model <dir> --max-samples 200

Log the ranked summary to the same W&B project training used, so the
training curves and the final test-set ranking sit side by side:
    python3 evaluation/evaluate_finetuned.py --model <dir> --lora <dir>:<base> \\
        --wandb-project whisper --run-name model-selection
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
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TARGET_SR = 16000

# Fixed, not a CLI flag: scripts and data are uploaded to the GPU pod
# together (see README.md), so there's no need to pass a path at run time.
TEST_PARQUET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stratified", "test.parquet")

NORMALIZE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def _read_parquet_table(path: str, columns: list[str]):
    """`memory_map` only applies to local files -- a `gs://...` path goes
    through pyarrow's GCS filesystem instead, which doesn't support it."""
    is_remote = path.startswith("gs://")
    return pq.read_table(path, columns=columns, memory_map=not is_remote)


class EvalAudioDataset(torch.utils.data.Dataset):
    """Minimal (audio_array, reference_text) reader -- no tokenization here,
    since each model under test brings its own processor/tokenizer."""

    def __init__(self, parquet_path, max_samples=None):
        self.table = _read_parquet_table(parquet_path, columns=["audio", "text"])
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


def load_model(spec, device):
    processor = WhisperProcessor.from_pretrained(spec["processor_repo"])
    if spec["type"] == "full":
        model = WhisperForConditionalGeneration.from_pretrained(spec["path"])
    elif spec["type"] == "lora":
        from peft import PeftModel
        base = WhisperForConditionalGeneration.from_pretrained(spec["base_model"])
        model = PeftModel.from_pretrained(base, spec["path"])
    else:
        raise ValueError(f"unknown model type: {spec['type']}")
    model.generation_config.language = "sinhala"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return processor, model.to(device).eval()


def evaluate_model(spec, dataset, device, batch_size):
    processor, model = load_model(spec, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_eval)

    predictions, references = [], []
    t0 = time.time()
    with torch.no_grad():
        for audios, texts in tqdm(loader, total=len(loader), desc=spec["path"], unit="batch"):
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
    parser.add_argument("--model", action="append", default=[], dest="models",
                         metavar="DIR",
                         help="a finetune_whisper.py --output-dir (full fine-tune) to score; repeatable")
    parser.add_argument("--lora", action="append", default=[], dest="loras",
                         metavar="DIR:BASE_MODEL",
                         help="a finetune_whisper_lora.py --output-dir, as '<adapter-dir>:<base-model>' "
                              "(e.g. /workspace/whisper-small-sinhala-lora/run1:openai/whisper-small); repeatable")
    parser.add_argument("--max-samples", type=int, default=None, help="subsample for a quick pass; omit for the full test set")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=".",
                         help="where to write per-model prediction CSVs (<model-name>_predictions.csv)")
    parser.add_argument("--wandb-project", default=None,
                         help="log the ranked summary table to this W&B project; omit to skip logging")
    parser.add_argument("--run-name", default=None, help="W&B run name, used only with --wandb-project")
    args = parser.parse_args()

    specs = {}
    for path in args.models:
        specs[path] = {"type": "full", "path": path, "processor_repo": path}
    for spec_str in args.loras:
        adapter_dir, _, base_model = spec_str.partition(":")
        if not base_model:
            raise SystemExit(f"--lora expects '<adapter-dir>:<base-model>', got: {spec_str!r}")
        specs[adapter_dir] = {"type": "lora", "path": adapter_dir, "base_model": base_model, "processor_repo": base_model}

    if not specs:
        parser.error("pass at least one --model and/or --lora")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading test set: {TEST_PARQUET}")
    dataset = EvalAudioDataset(TEST_PARQUET, max_samples=args.max_samples)
    print(f"{len(dataset)} rows\n")

    results = {}
    for name, spec in specs.items():
        print(f"=== {name} ({spec['type']}) ===")
        r = evaluate_model(spec, dataset, device, args.batch_size)
        results[name] = r
        print(f"  WER: {r['wer']:.2f}  CER: {r['cer']:.2f}  "
              f"({r['n_samples']} samples, {r['elapsed_s']:.0f}s)\n")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["wer"])

    print("=== Summary (best WER first) ===")
    print(f"{'model':<55} {'WER':>8} {'CER':>8}")
    for name, r in ranked:
        print(f"{name:<55} {r['wer']:>7.2f}% {r['cer']:>7.2f}%")

    best_name, best_r = ranked[0]
    print(f"\nBest model: {best_name}  (WER {best_r['wer']:.2f}%, CER {best_r['cer']:.2f}%)")

    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.run_name, job_type="eval")
        table = wandb.Table(columns=["model", "wer", "cer", "n_samples", "rank"])
        for rank, (name, r) in enumerate(ranked, start=1):
            table.add_data(name, r["wer"], r["cer"], r["n_samples"], rank)
            run.log({f"test/{name}/wer": r["wer"], f"test/{name}/cer": r["cer"]})
        run.log({"test/summary": table, "test/best_model": best_name, "test/best_wer": best_r["wer"]})
        run.finish()
        print(f"\nlogged ranked summary to W&B project '{args.wandb_project}'")

    os.makedirs(args.output_dir, exist_ok=True)
    for name, r in results.items():
        safe_name = name.strip("/").replace("/", "_")
        out_path = os.path.join(args.output_dir, f"{safe_name}_predictions.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reference", "prediction"])
            for ref, pred in zip(r["references"], r["predictions"]):
                writer.writerow([ref, pred])
        print(f"wrote per-sample predictions for {name} -> {out_path}")


if __name__ == "__main__":
    main()
