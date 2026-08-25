"""Full fine-tune of Whisper-small for Sinhala ASR (every weight unfrozen) on
the `stratified/` split of the final datasets. Meant to run on a GPU pod
(e.g. RunPod) -- tokenization (audio decode + log-mel feature extraction +
text tokenization) happens on the fly via `WhisperASRDataset` from
`prepare_whisper_dataset.py`, inside the `DataLoader` workers, overlapped
with GPU compute. There is no separate "tokenize first, then train" step:
running this script does both, and it avoids ever precomputing/storing the
~110GB a fully-cached spectrogram set for the 123,862-row train split would
take.

For a parameter-efficient alternative that trains a small LoRA adapter
instead of the full model, see `finetune_whisper_lora.py` -- same CLI shape,
same data pipeline, far less GPU memory and a much smaller artifact to save.

Per SinhaSpeech_Proporsal.pdf section 4.2:
  - stratified/train.parquet + stratified/validation.parquet for training
    and per-epoch WER/CER validation (best-checkpoint selection, not
    last-checkpoint)
  - Mixed precision (bf16 on Ampere+ GPUs, fp16 otherwise -- auto-detected)
    to cut memory use and speed up training
  - SpecAugment (time + frequency masking) enabled via WhisperConfig's
    built-in support (`apply_spec_augment`), active only in `model.train()`
    mode -- no hand-rolled masking code needed

`held_out/` is a secondary diagnostic split, not used here.

Dataset paths are NOT passed on the command line -- this script expects the
scripts and the data to be uploaded to the GPU pod together, so the paths
are fixed constants (TRAIN_PARQUET / EVAL_PARQUET below), relative to this
file's own directory:

    final-scripts/
      finetune_whisper.py          <- this file
      data/
        stratified/
          train.parquet
          validation.parquet
          test.parquet             <- used by evaluate_finetuned.py

See README.md for how to get the data into that layout (GCS download or
local copy) before running this script.

Usage (on a RunPod GPU pod, or any machine with a GPU -- run from inside
final-scripts/, with data/stratified/ already populated):
    python3 finetune_whisper.py \\
        --output-dir /workspace/whisper-small-sinhala/run6-lr3e-5-bs32 \\
        --run-name run6-lr3e-5-bs32 \\
        --wandb-project whisper \\
        --learning-rate 3e-5 \\
        --per-device-train-batch-size 32 \\
        --num-train-epochs 4

Smoke test on CPU (no GPU needed, a handful of steps, sanity-checks the
pipeline without a real training run):
    python3 finetune_whisper.py --smoke-test
"""

import argparse
import os
import sys

import evaluate
import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(__file__))
from prepare_whisper_dataset import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    WhisperASRDataset,
    build_processor,
)

# Fixed, not CLI flags: scripts and data are uploaded to the GPU pod together
# (see README.md), so there's no need to pass paths at run time.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stratified")
TRAIN_PARQUET = os.path.join(DATA_DIR, "train.parquet")
EVAL_PARQUET = os.path.join(DATA_DIR, "validation.parquet")

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def pick_mixed_precision():
    """bf16 on Ampere+ (no loss-scaling headaches), fp16 elsewhere, fp32 on CPU."""
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    if torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}
    return {"fp16": True, "bf16": False}


def enable_spec_augment(model, time_mask_prob=0.05, feature_mask_prob=0.05):
    """WhisperConfig ships SpecAugment support (same mixin as Wav2Vec2) --
    it's off by default. Feature masking defaults to 0.0 upstream (time-only);
    the proposal asks for time *and* frequency masking, so both are set here.
    """
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = time_mask_prob
    model.config.mask_feature_prob = feature_mask_prob


def make_compute_metrics(processor):
    pad_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = pad_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", default="whisper-small-sinhala")
    parser.add_argument("--run-name", default=None,
                         help="W&B run name; defaults to the --output-dir basename")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler-type", default="cosine",
                         help="passed straight to Seq2SeqTrainingArguments -- 'cosine', 'linear', "
                              "'constant_with_warmup', etc. (see transformers SchedulerType)")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--no-spec-augment", action="store_true")
    parser.add_argument("--noise-prob", type=float, default=0.2,
                         help="probability per training sample of mixing in light Gaussian noise (0 to disable)")
    parser.add_argument("--noise-snr-db-min", type=float, default=20.0)
    parser.add_argument("--noise-snr-db-max", type=float, default=30.0)
    parser.add_argument("--stretch-prob", type=float, default=0.2,
                         help="probability per training sample of time-stretching (0 to disable)")
    parser.add_argument("--stretch-rate-min", type=float, default=0.9)
    parser.add_argument("--stretch-rate-max", type=float, default=1.1)
    parser.add_argument("--pitch-prob", type=float, default=0.2,
                         help="probability per training sample of pitch-shifting (0 to disable)")
    parser.add_argument("--pitch-semitones-min", type=float, default=-2.0)
    parser.add_argument("--pitch-semitones-max", type=float, default=2.0)
    parser.add_argument("--wandb-project", default=None, help="omit to disable W&B logging")
    parser.add_argument("--smoke-test", action="store_true",
                         help="tiny CPU run (a few steps, no generation-based eval) to sanity-check the pipeline")
    args = parser.parse_args()

    print(f"Loading processor + model: {args.model_name}")
    processor = build_processor(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    model.generation_config.language = "sinhala"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if not args.no_spec_augment:
        enable_spec_augment(model)
        print(f"SpecAugment enabled: mask_time_prob={model.config.mask_time_prob}, "
              f"mask_feature_prob={model.config.mask_feature_prob}")

    print(f"Loading train split: {TRAIN_PARQUET}")
    train_dataset = WhisperASRDataset(
        TRAIN_PARQUET,
        processor,
        noise_prob=args.noise_prob,
        noise_snr_db=(args.noise_snr_db_min, args.noise_snr_db_max),
        stretch_prob=args.stretch_prob,
        stretch_rate_range=(args.stretch_rate_min, args.stretch_rate_max),
        pitch_prob=args.pitch_prob,
        pitch_semitone_range=(args.pitch_semitones_min, args.pitch_semitones_max),
    )
    if args.noise_prob > 0:
        print(f"Noise augmentation enabled: prob={args.noise_prob}, "
              f"snr_db=({args.noise_snr_db_min}, {args.noise_snr_db_max})")
    if args.stretch_prob > 0:
        print(f"Time-stretch augmentation enabled: prob={args.stretch_prob}, "
              f"rate=({args.stretch_rate_min}, {args.stretch_rate_max})")
    if args.pitch_prob > 0:
        print(f"Pitch-shift augmentation enabled: prob={args.pitch_prob}, "
              f"semitones=({args.pitch_semitones_min}, {args.pitch_semitones_max})")
    print(f"Loading eval split: {EVAL_PARQUET}")
    eval_dataset = WhisperASRDataset(EVAL_PARQUET, processor)  # no noise -- eval must stay on clean audio
    print(f"train rows: {len(train_dataset)}  eval rows: {len(eval_dataset)}")

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    precision = pick_mixed_precision()
    print(f"Mixed precision: {precision}")

    if args.smoke_test:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(4, len(train_dataset))))
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(min(4, len(eval_dataset))))
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=2,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=1,
            predict_with_generate=False,
            remove_unused_columns=False,
            report_to=[],
            dataloader_num_workers=0,
            **precision,
        )
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            processing_class=processor,
        )
        print("\nRunning smoke test (2 steps, no generation-based eval)...")
        trainer.train()
        print("\nSmoke test passed -- pipeline runs end to end.")
        return

    run_name = args.run_name or os.path.basename(os.path.normpath(args.output_dir))
    report_to = ["wandb"] if args.wandb_project else []

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name,
        **precision,
    )
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving best checkpoint to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    print(f"\nFinal validation metrics: {metrics}")


if __name__ == "__main__":
    main()
