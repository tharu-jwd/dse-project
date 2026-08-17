"""Turn a `final_split_dataset/*/*.parquet` file into a PyTorch `Dataset` that
yields Whisper-ready `(input_features, labels)` pairs -- log-mel spectrograms
from the `audio` column, tokenized Sinhala transcripts from `text`.

Why on-the-fly, not precomputed: log-mel features are float32 and always
padded/trimmed to Whisper's fixed 30s window (80 x 3000 per example, ~937KB),
regardless of the clip's real length. Precomputing and caching that for
`stratified/train.parquet` (123,862 rows, mostly 2-5s OpenSLR clips) would
balloon to ~110GB on disk -- infeasible given this machine's free space. This
module instead decodes audio and extracts features lazily in `__getitem__`,
so nothing beyond one batch is ever materialized. The tradeoff is CPU work
repeated every epoch; if that becomes the bottleneck, increase
`DataLoader(num_workers=...)` before reaching for a precomputed cache.

Usage as a library (typical Seq2SeqTrainer setup):

    from prepare_whisper_dataset import WhisperASRDataset, DataCollatorSpeechSeq2SeqWithPadding, build_processor

    processor = build_processor("openai/whisper-small")
    train_ds = WhisperASRDataset("dse-project/model-development/data/final_split_dataset/stratified/train.parquet", processor)
    eval_ds = WhisperASRDataset("dse-project/model-development/data/final_split_dataset/stratified/validation.parquet", processor)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # pass train_ds / eval_ds / collator straight to Seq2SeqTrainer

`parquet_path` may also be a `gs://bucket/.../train.parquet` URI -- the
datasets are staged in a GCS bucket for cloud GPU runs. Requires
`GOOGLE_APPLICATION_CREDENTIALS` (or ADC/workload identity on the pod) to be
set; no bucket auth handling lives in this module.

Usage as a smoke test (from repo root):

    python3 dse-project/scripts/prepare_whisper_dataset.py \\
        dse-project/model-development/data/final_split_dataset/stratified/test.parquet
    python3 dse-project/scripts/prepare_whisper_dataset.py \\
        gs://sinhaspeech-data/final_split_dataset/stratified/test.parquet
"""

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperTokenizer

LANGUAGE = "sinhala"
TASK = "transcribe"
TARGET_SR = 16000


def _read_parquet_table(path: str, columns: list[str]):
    """`pq.read_table`, but `memory_map` only applies to local files -- a
    `gs://...` path (our datasets live in a GCS bucket for cloud GPU runs)
    goes through pyarrow's GCS filesystem instead, which doesn't support it.
    Requires `GOOGLE_APPLICATION_CREDENTIALS` (or workload-identity/ADC) to
    be set up for bucket access; no code-level auth handling needed here.
    """
    is_remote = path.startswith("gs://")
    return pq.read_table(path, columns=columns, memory_map=not is_remote)


def build_processor(model_name: str = "openai/whisper-small") -> WhisperProcessor:
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=LANGUAGE, task=TASK)
    return WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


class WhisperASRDataset(torch.utils.data.Dataset):
    """Wraps one split parquet file (schema: audio/source_dataset/text).

    Local paths read via `pyarrow.parquet.read_table(memory_map=True)`, so the
    OS page cache backs the audio bytes instead of a full Python-heap copy --
    opening even the 14GB `stratified/train.parquet` is cheap; only accessed
    rows get paged in. `gs://...` paths go through pyarrow's GCS filesystem
    instead (no memory-mapping there, but still one HTTP range read per row
    rather than downloading the whole file up front).
    """

    def __init__(self, parquet_path: str, processor: WhisperProcessor):
        self.table = _read_parquet_table(parquet_path, columns=["audio", "text"])
        self.processor = processor

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        audio_bytes = self.table.column("audio")[idx].as_py()
        text = self.table.column("text")[idx].as_py()

        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR
        ).input_features[0]
        labels = self.processor.tokenizer(text).input_ids

        return {
            "input_features": np.asarray(input_features, dtype=np.float32),
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Standard Whisper fine-tuning collator (per the HF Whisper blog recipe):
    pads `input_features` and `labels` independently, masks label padding with
    -100 so it's ignored by the loss, and strips a leading BOS token if the
    tokenizer already added one (Trainer/generate adds its own at decode time).
    """

    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def _smoke_test(parquet_path: str, model_name: str, n: int):
    import time

    print(f"Building processor for {model_name} (language={LANGUAGE}, task={TASK})...")
    processor = build_processor(model_name)

    print(f"Opening {parquet_path} ...")
    ds = WhisperASRDataset(parquet_path, processor)
    print(f"{len(ds)} rows")

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    t0 = time.time()
    items = [ds[i] for i in range(min(n, len(ds)))]
    elapsed = time.time() - t0
    print(f"processed {len(items)} rows in {elapsed:.2f}s ({elapsed / len(items):.3f}s/row)")

    for i, item in enumerate(items[:3]):
        text_back = processor.tokenizer.decode(item["labels"], skip_special_tokens=True)
        print(f"  [{i}] input_features shape={item['input_features'].shape}  "
              f"n_labels={len(item['labels'])}  text={text_back!r}")

    batch = collator(items)
    print(f"\ncollated batch: input_features={tuple(batch['input_features'].shape)}  "
          f"labels={tuple(batch['labels'].shape)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet_path", help="a split file, e.g. .../stratified/test.parquet")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--n", type=int, default=8, help="rows to smoke-test")
    args = parser.parse_args()

    _smoke_test(args.parquet_path, args.model, args.n)
