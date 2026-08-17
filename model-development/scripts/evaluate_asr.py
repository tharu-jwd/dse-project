"""Evaluate ASR models on SPEAK-ASR's Sinhala test set and report WER.

Usage:
  python3 model-development/scripts/evaluate_asr.py --model speak-asr --num-samples 20
  python3 model-development/scripts/evaluate_asr.py --model plain-whisper --whisper-size medium --num-samples 20
  python3 model-development/scripts/evaluate_asr.py --model custom --checkpoint model-development/checkpoints/<run> --num-samples 20

`speak-asr` loads openai/whisper-medium + the SPEAK-ASR LoRA adapter
(SPEAK-ASR/whisper-si-exp-10-medium-all, documented eval WER 10.85%).
`plain-whisper` loads a stock (non-fine-tuned) openai-whisper model for
comparison. `custom` loads a locally fine-tuned LoRA checkpoint (e.g. one
produced by train_asr.py) via --checkpoint.

Every run appends a result row to eval_results/results.jsonl unless
--no-persist is given.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from datasets import Audio, load_dataset

# asr_common is a sibling of this scripts/ dir, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_common.metrics import collect_predictions, compute_wer
from asr_common.models import get_device, load_lora_adapter, load_plain_whisper

DATASET = "SPEAK-ASR/openslr-sinhala-asr"
SPEAK_ASR_BASE = "openai/whisper-medium"
SPEAK_ASR_ADAPTER = "SPEAK-ASR/whisper-si-exp-10-medium-all"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval_results" / "results.jsonl"


def get_git_commit() -> str | None:
    """Short commit hash of the code that produced a result, for provenance. None outside git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def append_eval_result(record: dict) -> None:
    """Append one JSON line to eval_results/results.jsonl (JSONL: easy to append, easy to diff)."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["speak-asr", "plain-whisper", "custom"], required=True)
    parser.add_argument("--whisper-size", default="medium", help="only used with --model plain-whisper")
    parser.add_argument("--checkpoint", default=None, help="local path or Hub id; required with --model custom")
    parser.add_argument("--base-model", default=SPEAK_ASR_BASE, help="only used with --model custom")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-persist", action="store_true", help="don't log this run to eval_results/results.jsonl")
    args = parser.parse_args()

    if args.model == "custom" and not args.checkpoint:
        parser.error("--model custom requires --checkpoint")
    if args.model != "custom" and args.checkpoint:
        parser.error("--checkpoint is only used with --model custom")
    return args


def main():
    args = parse_args()
    device = get_device()

    print(f"Loading model ({args.model}) on {device}...")
    base_model = None
    if args.model == "speak-asr":
        base_model = SPEAK_ASR_BASE
        transcribe = load_lora_adapter(base_model, SPEAK_ASR_ADAPTER, device)
    elif args.model == "custom":
        base_model = args.base_model
        transcribe = load_lora_adapter(base_model, args.checkpoint, device)
    else:
        transcribe = load_plain_whisper(args.whisper_size, device)

    ds = load_dataset(DATASET, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    refs, hyps = collect_predictions(transcribe, ds, args.num_samples)

    wer = compute_wer(refs, hyps)
    print(f"\n=== Overall WER over {len(refs)} samples: {wer * 100:.2f}% ===")

    if not args.no_persist:
        append_eval_result(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "checkpoint": args.checkpoint,
                "base_model": base_model,
                "dataset": DATASET,
                "split": args.split,
                "num_samples": len(refs),
                "wer": wer,
                "git_commit": get_git_commit(),
            }
        )


if __name__ == "__main__":
    main()
