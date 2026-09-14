from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("training_entrypoint", ROOT / "scripts/training/train.py")
assert SPEC is not None and SPEC.loader is not None
TRAIN = module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)


class FakeFeatureExtractor:
    def __call__(self, audio, **kwargs):
        del kwargs
        return {"input_features": torch.tensor(audio), "attention_mask": torch.ones(2, 2)}


class FakeTokenizer:
    bos_token_id = 1

    def __init__(self) -> None:
        self.language = "si"
        self.prefix_calls: list[str] = []

    def set_prefix_tokens(self, *, language: str, task: str) -> None:
        assert task == "transcribe"
        self.language = language
        self.prefix_calls.append(language)

    def __call__(self, text: str):
        language_token = 10 if self.language == "si" else 20
        return SimpleNamespace(input_ids=[1, language_token, len(text)])

    def pad(self, rows, *, return_tensors: str):
        assert return_tensors == "pt"
        ids = torch.tensor([row["input_ids"] for row in rows])
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_collator_uses_each_rows_decoder_language_and_restores_sinhala() -> None:
    tokenizer = FakeTokenizer()
    processor = SimpleNamespace(
        feature_extractor=FakeFeatureExtractor(), tokenizer=tokenizer
    )
    collator = TRAIN.WhisperCollator(processor)

    batch = collator(
        [
            {"audio": [0.0, 0.0], "text": "සිංහල", "decoder_language": "si"},
            {"audio": [0.0, 0.0], "text": "English", "decoder_language": "en"},
        ]
    )

    assert tokenizer.prefix_calls == ["si", "en", "si"]
    assert tokenizer.language == "si"
    assert batch["labels"].tolist() == [[10, 5], [20, 7]]
