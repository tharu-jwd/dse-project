"""Fine-tune Whisper on Sinhala speech with a LoRA adapter.

Usage (real run, cloud GPU):
  python3 model-development/scripts/train_asr.py \\
    --output-dir model-development/checkpoints/whisper-medium-si-lora --num-train-epochs 3

Usage (local smoke test, no GPU needed; proves the pipeline runs, not that
the resulting model is any good):
  python3 model-development/scripts/train_asr.py \\
    --streaming --max-steps 5 --max-train-samples 20 --max-eval-samples 5 \\
    --per-device-train-batch-size 1 --eval-steps 5 --save-steps 5 \\
    --output-dir model-development/checkpoints/smoke-test

Trains a LoRA adapter on top of a frozen Whisper base model and saves it as
a standard PEFT adapter directory, the same shape asr_common.models.load_lora_adapter
already knows how to load, so the result can be scored with evaluate_asr.py
(--model custom --checkpoint <output-dir>) or loaded by anything else that
speaks PEFT.

Training data comes from asr_common.dev_dataset, a temporary placeholder
until the real preprocessing pipeline is ready (see that module's docstring).
"""

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from peft import get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from asr_common.dev_dataset import load_dev_openslr_split
from asr_common.metrics import compute_wer
from asr_common.models import build_lora_config, get_device, load_processor_and_base_model


def build_preprocess_fn(processor):
    """Turn one raw {audio, text} example into Whisper's expected input/label arrays.

    Audio arrives as undecoded bytes (see dev_dataset.py); decode with
    soundfile here, same as evaluate_asr.py does, rather than relying on
    datasets' auto-decode (which pulls in torchcodec).
    """

    def preprocess(example):
        array, sr = sf.read(io.BytesIO(example["audio"]["bytes"]))
        input_features = processor.feature_extractor(array, sampling_rate=sr).input_features[0]
        labels = processor.tokenizer(example["text"]).input_ids
        return {"input_features": input_features, "labels": labels}

    return preprocess


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads a batch of variable-length features/labels for Seq2SeqTrainer.

    Audio features and text labels are padded independently since they're
    unrelated sequences with unrelated lengths.
    """

    processor: object

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # -100 tells the loss to ignore these positions: padding isn't a real token.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # The tokenizer prepends its own start token, and Whisper's decoder adds
        # one too during generation, so drop the tokenizer's here to avoid a duplicate.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_compute_metrics(processor):
    """WER-during-training, using the same compute_wer() evaluate_asr.py uses."""

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        # Reverse the -100 masking before decoding: batch_decode doesn't
        # know about it, that's a Trainer/loss-only convention.
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": compute_wer(label_str, pred_str)}

    return compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default="openai/whisper-medium")
    parser.add_argument("--output-dir", default="model-development/checkpoints/whisper-medium-si-lora")
    parser.add_argument("--run-name", default=None, help="appended to --output-dir to keep experiments separate")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--streaming", action="store_true", help="stream the dataset instead of downloading it fully")
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="required when --streaming")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    args = parser.parse_args()

    if args.streaming and args.max_steps < 0:
        parser.error("--streaming requires --max-steps (a streamed dataset has no known length)")
    return args


def resolve_precision(requested: str, device: str) -> str:
    if requested == "auto":
        return "fp16" if device == "cuda" else "fp32"
    # Whisper + fp16/bf16 on MPS or CPU is a known source of NaN losses,
    # so only cuda gets to opt into reduced precision.
    if requested in ("fp16", "bf16") and device != "cuda":
        raise SystemExit(f"--precision {requested} is only supported on cuda (device is {device})")
    return requested


def main():
    args = parse_args()
    device = get_device()
    precision = resolve_precision(args.precision, device)

    output_dir = str(Path(args.output_dir) / args.run_name) if args.run_name else args.output_dir
    train_batch_size = args.per_device_train_batch_size or (16 if device == "cuda" else 2)
    eval_batch_size = args.per_device_eval_batch_size or (16 if device == "cuda" else 2)

    print(f"Loading base model ({args.base_model}) on {device}, precision={precision}...")
    processor, base_model = load_processor_and_base_model(args.base_model, device)

    target_modules = args.lora_target_modules.split(",")
    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, target_modules)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_ds = load_dev_openslr_split("train", streaming=args.streaming)
    eval_ds = load_dev_openslr_split(args.eval_split, streaming=args.streaming)

    if args.max_train_samples:
        train_ds = train_ds.take(args.max_train_samples) if args.streaming else train_ds.select(range(args.max_train_samples))
    if args.max_eval_samples:
        eval_ds = eval_ds.take(args.max_eval_samples) if args.streaming else eval_ds.select(range(args.max_eval_samples))

    preprocess = build_preprocess_fn(processor)
    # Whatever columns the source dataset happens to have, not hardcoded,
    # so swapping in the real preprocessed dataset later doesn't break this.
    original_columns = train_ds.column_names
    train_ds = train_ds.map(preprocess, remove_columns=original_columns)
    eval_ds = eval_ds.map(preprocess, remove_columns=original_columns)

    if args.streaming:
        train_ds = train_ds.shuffle(buffer_size=500, seed=42)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        fp16=(precision == "fp16"),
        bf16=(precision == "bf16"),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=args.report_to,
        # A wrapped PeftModel confuses Trainer's automatic column/label
        # detection, so both of these have to be set explicitly.
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Saving a PeftModel writes only the adapter (config + weights), not the
    # multi-gigabyte base model, the same shape load_lora_adapter() expects.
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
