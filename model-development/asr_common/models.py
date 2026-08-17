"""Shared helpers for loading Whisper models and LoRA adapters.

Used by evaluate_asr.py (scoring the SPEAK-ASR baseline, stock Whisper, or a
locally-trained checkpoint) and by train_asr.py (loading the base model to
fine-tune, and optionally scoring its own checkpoint the same way
evaluate_asr.py would). Routing through load_lora_adapter either way keeps
the two scripts from ever disagreeing on how a model gets loaded or run.

Heavy ML libraries (transformers, peft, openai-whisper) are imported inside
the functions that need them, not at module load time, so importing this
module doesn't force-load libraries a given script run won't actually use.
"""

import numpy as np
import torch


def get_device() -> str:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_processor_and_base_model(model_id: str, device: str):
    """Load a Whisper processor + base model and move the model to `device`."""
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    return processor, model


def build_lora_config(r: int, alpha: int, dropout: float, target_modules: list[str]):
    """LoRA config for training a fresh adapter on top of Whisper's attention projections."""
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, bias="none")


def get_forced_decoder_ids(processor, language: str = "si", task: str = "transcribe"):
    """Decoder prompt that forces Whisper to transcribe (not translate) Sinhala."""
    return processor.get_decoder_prompt_ids(language=language, task=task)


def load_lora_adapter(base_model_id: str, adapter_path_or_id: str, device: str):
    """Load a base Whisper model with a LoRA adapter applied, ready for inference.

    `adapter_path_or_id` can be a local checkpoint directory (e.g. one written
    by train_asr.py) or a Hub adapter id (e.g. the SPEAK-ASR baseline).
    PeftModel.from_pretrained handles both the same way.

    Returns a `transcribe(array, sr) -> str` closure, same shape as
    load_plain_whisper's, so callers don't need to care which one they got.
    """
    from peft import PeftModel

    processor, base_model = load_processor_and_base_model(base_model_id, device)
    model = PeftModel.from_pretrained(base_model, adapter_path_or_id)
    model.to(device)
    model.eval()
    forced_decoder_ids = get_forced_decoder_ids(processor)

    def transcribe(array, sr):
        inputs = processor(array, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=440,
            )
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return transcribe


def load_plain_whisper(size: str, device: str):
    """Load a stock (non-fine-tuned) openai-whisper model, for baseline comparison."""
    import whisper

    model = whisper.load_model(size, device=device)

    def transcribe(array, sr):
        result = model.transcribe(array.astype(np.float32), language="si", fp16=False)
        return result["text"].strip()

    return transcribe
