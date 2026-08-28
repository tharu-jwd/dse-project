"""Convert the fine-tuned Whisper checkpoint to CTranslate2 format.

faster-whisper (the streaming inference engine) requires a CTranslate2
model, not a raw Hugging Face checkpoint. Run this once, whenever the
checkpoint changes, before enabling streaming:

    PYTHONPATH=. .venv/bin/python scripts/convert_to_ctranslate2.py

If STREAMING_SOURCE_MODEL is a base model + adapter is used instead of a
complete checkpoint, pass --adapter to merge a LoRA adapter first.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import ctranslate2.converters

from app.core.config import settings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Whisper checkpoint to CTranslate2 format "
        "for faster-whisper streaming inference.",
    )
    parser.add_argument(
        "--source",
        default=settings.streaming_model_source_path,
        help="Complete Hugging Face Whisper checkpoint (local path or Hub ID).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT/LoRA adapter to merge into --source before converting.",
    )
    parser.add_argument(
        "--output",
        default=settings.streaming_model_ct2_path,
        help="Output directory for the converted CTranslate2 model.",
    )
    parser.add_argument(
        "--quantization",
        default=settings.streaming_compute_type,
        help="CTranslate2 quantization/compute type, e.g. int8, float16.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def merge_adapter(source: str, adapter: str, merge_dir: Path) -> str:
    """Merge a PEFT/LoRA adapter into its base model and save the result."""

    from peft import PeftModel
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    print(f"Loading base model {source} and adapter {adapter} for merging...")
    base_model = WhisperForConditionalGeneration.from_pretrained(source)
    processor = AutoProcessor.from_pretrained(source)

    merged = PeftModel.from_pretrained(base_model, adapter)
    merged = merged.merge_and_unload()

    merge_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merge_dir)
    processor.save_pretrained(merge_dir)

    print(f"Merged checkpoint written to {merge_dir}")
    return str(merge_dir)


def main() -> None:
    arguments = parse_arguments()
    output_path = Path(arguments.output)

    if output_path.exists():
        if not arguments.force:
            raise SystemExit(
                f"Output directory already exists: {output_path}. "
                "Pass --force to overwrite it."
            )
        shutil.rmtree(output_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        source = arguments.source

        if arguments.adapter:
            source = merge_adapter(
                arguments.source,
                arguments.adapter,
                Path(temp_dir) / "merged",
            )

        print(f"Converting {source} -> {output_path} ({arguments.quantization})")
        converter = ctranslate2.converters.TransformersConverter(source)
        converter.convert(
            str(output_path),
            quantization=arguments.quantization,
        )

    print("Done. Set STREAMING_MODEL_PATH accordingly if you used --output.")


if __name__ == "__main__":
    main()
