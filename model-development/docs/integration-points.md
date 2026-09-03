# Backend integration points

The backend consumes trained ASR models through the Transcriber protocol in
`backend/app/transcribers/base.py`. The protocol returns transcript text and
timestamped segments without coupling application code to a particular model.

`backend/app/transcribers/factory.py` currently supports three implementations:

- `fake`: canned output for lightweight local development
- `whisper`: a complete local or Hugging Face Whisper checkpoint
- `speak_asr`: a PEFT adapter loaded over a configured Whisper base model

Select the implementation with `TRANSCRIBER_BACKEND`. The related model source,
base model, adapter, and language settings are documented in `backend/README.md`
and the repository `.env.example`.

The training workflow produces either a complete checkpoint through
`training/finetune_whisper.py` or a PEFT adapter through
`training/finetune_whisper_lora.py`. Point the matching backend configuration at
that output after it has passed the fixed Sinhala test set and application
latency checks.

Dataset preparation is upstream of the backend. Training scripts read
`model-development/data/stratified/train.parquet` and `validation.parquet`,
evaluation reads `test.parquet`, and model artifacts are deployed separately
rather than committed to Git.
