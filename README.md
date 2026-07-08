# Sinhala ASR — Domain-Specific Speech Transcription

Development of a Sinhala Automatic Speech Recognition (ASR) system: collecting/curating
speech data, preprocessing audio, fine-tuning a model (Whisper), evaluating transcription
accuracy with WER, and building a simple record/transcribe interface.

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
cd whisper-live-test
python3 server.py
```

Open http://localhost:5005, pick a model (use "SPEAK-ASR (Sinhala fine-tuned)" for Sinhala),
and record or upload audio.

## Running an evaluation

```
python3 scripts/evaluate_asr.py --model speak-asr --num-samples 20
python3 scripts/evaluate_asr.py --model plain-whisper --whisper-size medium --num-samples 20
```

## Setup

Requires Python 3.11+, `ffmpeg` on PATH, and:

```
pip3 install flask openai-whisper transformers torch datasets soundfile peft jiwer
```
