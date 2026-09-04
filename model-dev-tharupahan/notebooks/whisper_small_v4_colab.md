# Whisper-small v4 validation baseline (Google Colab)

This is the minimal Colab procedure. It performs inference only on the 206
audio-verified v4 validation clips. It does not train and never opens the test
split.

1. Create a new Colab notebook and select **Runtime → Change runtime type → T4 GPU**.
2. Upload `v4-validation-colab.parquet` when the upload cell asks for it.
3. Run the following cells in order.

```python
!pip -q install "transformers>=4.46" accelerate pyarrow soundfile
```

```python
import torch
assert torch.cuda.is_available(), "Enable a GPU runtime before continuing"
print(torch.cuda.get_device_name(0))
```

```python
from google.colab import files
uploaded = files.upload()
assert "v4-validation-colab.parquet" in uploaded
```

```python
import io, json, time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = "openai/whisper-small"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 225
rows = pq.read_table("v4-validation-colab.parquet").to_pylist()
assert len(rows) == 206

processor = WhisperProcessor.from_pretrained(MODEL, language="si", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()
model.generation_config.language = "si"
model.generation_config.task = "transcribe"

predictions = []
started = time.monotonic()
with torch.inference_mode():
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        audio = []
        for row in batch:
            waveform, rate = sf.read(io.BytesIO(row["audio"]), dtype="float32")
            assert rate == 16000 and waveform.ndim == 1
            audio.append(waveform)
        features = processor.feature_extractor(
            audio, sampling_rate=16000, return_attention_mask=True, return_tensors="pt"
        )
        generated = model.generate(
            features.input_features.to("cuda", dtype=torch.float16),
            attention_mask=features.attention_mask.to("cuda"),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        predictions.extend(processor.tokenizer.batch_decode(generated, skip_special_tokens=True))
        print(f"{min(start + BATCH_SIZE, len(rows))}/{len(rows)}")

result = []
for row, prediction in zip(rows, predictions):
    result.append({k: v for k, v in row.items() if k != "audio"} | {
        "prediction": prediction.strip(), "model": MODEL
    })
pq.write_table(pa.Table.from_pylist(result), "whisper-small-v4-validation-predictions.parquet")
Path("whisper-small-v4-validation-runtime.json").write_text(json.dumps({
    "model": MODEL,
    "rows": len(result),
    "gpu": torch.cuda.get_device_name(0),
    "batch_size": BATCH_SIZE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "runtime_seconds": time.monotonic() - started,
}, indent=2) + "\n")
```

```python
from google.colab import files
files.download("whisper-small-v4-validation-predictions.parquet")
files.download("whisper-small-v4-validation-runtime.json")
```

Return both downloaded files. They are scored locally using the repository's
versioned strict/canonical metric and detailed error-analysis implementation.
