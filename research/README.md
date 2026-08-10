# Sinhala ASR Research

Early-stage research into a Sinhala Automatic Speech Recognition (ASR) system: evaluating
existing models, fine-tuning candidates, and a throwaway record/transcribe demo used to
sanity-check them. This predates and informs the `backend`/`frontend` app (see the
[root README](../README.md)) — the real transcriber isn't wired into that app yet, and this
folder is where that integration work will start.

## Status

Evaluated existing prior work from the [SPEAK-ASR](https://huggingface.co/SPEAK-ASR)
HuggingFace org before starting our own fine-tuning:

- `SPEAK-ASR/whisper-medium-si-merged` — undocumented merged checkpoint, produces garbage
  output on most inputs. Not usable.
- `SPEAK-ASR/whisper-si-exp-10-medium-all` — a LoRA adapter on `openai/whisper-medium`,
  documented eval WER 10.85% / eval loss 0.199. Verified independently: produces accurate
  transcriptions and matches the claimed WER on spot checks.

This adapter is wired into the demo app as a working baseline to build on.

## Layout

- `whisper-live-test/` — Flask + browser demo: record or upload speech, get a transcription.
  Supports plain Whisper (`tiny`...`large-v3`) and the SPEAK-ASR Sinhala adapter.
- `scripts/evaluate_asr.py` — runs a model against SPEAK-ASR's Sinhala test set and reports WER.

## Running the demo

```
cd research/whisper-live-test
python3 server.py
```

Open http://localhost:5005, pick a model (use "SPEAK-ASR (Sinhala fine-tuned)" for Sinhala),
and record or upload audio.

## Running an evaluation

```
python3 research/scripts/evaluate_asr.py --model speak-asr --num-samples 20
python3 research/scripts/evaluate_asr.py --model plain-whisper --whisper-size medium --num-samples 20
```

## Setup

Requires Python 3.11+, `ffmpeg` on PATH, and:

```
pip3 install flask openai-whisper transformers torch datasets soundfile peft jiwer
```

## Models & data (not stored in this repo)

Model weights and datasets are downloaded automatically the first time you run the demo
or a script — there is nothing to manually download or place inside the repo. They're
fetched by the `transformers`/`huggingface_hub`/`whisper` libraries and cached outside
the project folder:

| What | Source | Cached at |
|---|---|---|
| Base Whisper (`medium`, etc., used by the demo's plain-Whisper models) | [openai/whisper](https://github.com/openai/whisper) | `~/.cache/whisper/` |
| `openai/whisper-medium` (HF copy, base for the adapter) | [huggingface.co/openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | `~/.cache/huggingface/hub/` |
| SPEAK-ASR Sinhala LoRA adapter | [huggingface.co/SPEAK-ASR/whisper-si-exp-10-medium-all](https://huggingface.co/SPEAK-ASR/whisper-si-exp-10-medium-all) | `~/.cache/huggingface/hub/` |
| Sinhala test set (for `evaluate_asr.py`) | [huggingface.co/datasets/SPEAK-ASR/openslr-sinhala-asr](https://huggingface.co/datasets/SPEAK-ASR/openslr-sinhala-asr) | `~/.cache/huggingface/hub/` |

`.gitignore` excludes `*.pt`/`*.safetensors`/`*.ckpt` so these never accidentally get
committed. If you need to free disk space, it's safe to delete either cache folder —
everything re-downloads on next run.
