"""Evaluate ASR models on SPEAK-ASR's Sinhala test set and report WER.

Usage:
  python3 scripts/evaluate_asr.py --model speak-asr --num-samples 20
  python3 scripts/evaluate_asr.py --model plain-whisper --whisper-size medium --num-samples 20

`speak-asr` loads openai/whisper-medium + the SPEAK-ASR LoRA adapter
(SPEAK-ASR/whisper-si-exp-10-medium-all, documented eval WER 10.85%).
`plain-whisper` loads a stock (non-fine-tuned) openai-whisper model for
comparison.
"""

import argparse
import io

import jiwer
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

DATASET = "SPEAK-ASR/openslr-sinhala-asr"
SPEAK_ASR_BASE = "openai/whisper-medium"
SPEAK_ASR_ADAPTER = "SPEAK-ASR/whisper-si-exp-10-medium-all"


def load_speak_asr():
    from peft import PeftModel
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    processor = AutoProcessor.from_pretrained(SPEAK_ASR_BASE)
    base_model = WhisperForConditionalGeneration.from_pretrained(SPEAK_ASR_BASE)
    model = PeftModel.from_pretrained(base_model, SPEAK_ASR_ADAPTER)
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="si", task="transcribe")

    def transcribe(array, sr):
        import torch

        inputs = processor(array, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=440,
            )
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return transcribe


def load_plain_whisper(size):
    import whisper

    model = whisper.load_model(size)

    def transcribe(array, sr):
        result = model.transcribe(array.astype(np.float32), language="si", fp16=False)
        return result["text"].strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["speak-asr", "plain-whisper"], required=True)
    parser.add_argument("--whisper-size", default="medium", help="only used with --model plain-whisper")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print(f"Loading model ({args.model})...")
    transcribe = load_speak_asr() if args.model == "speak-asr" else load_plain_whisper(args.whisper_size)

    ds = load_dataset(DATASET, split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    refs, hyps = [], []
    for i, sample in enumerate(ds.take(args.num_samples)):
        ref = sample["text"]
        array, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]))
        hyp = transcribe(array, sr)
        refs.append(ref)
        hyps.append(hyp)
        print(f"\n--- sample {i + 1} ---")
        print(f"REF: {ref}")
        print(f"HYP: {hyp}")

    wer = jiwer.wer(refs, hyps)
    print(f"\n=== Overall WER over {len(refs)} samples: {wer * 100:.2f}% ===")


if __name__ == "__main__":
    main()
