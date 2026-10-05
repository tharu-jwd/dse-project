# SinhaSpeech

An accessible platform for Sinhala-speaking students and teachers: live captioning of lectures,
recorded/uploaded transcription with transcript review and editing, speech-based quizzes and study
notes, and a hands-free voice command mode. It serves Deaf and hard-of-hearing students (who need
the teacher's speech as text) and students with writing difficulties (who answer by speaking).

The project has two phases: **Phase 1** fine-tunes Whisper-small for Sinhala (best test WER
**16.38%**), and **Phase 2** is the app built on it.

## Layout

- `backend/`: FastAPI + PostgreSQL API for auth, media upload, transcription job queue/worker,
  transcript CRUD, live streaming transcription, and voice command recognition. See
  [`backend/README.md`](backend/README.md).
- `frontend/`: React app for lectures, spoken quizzes, study notes, transcript review, and
  teacher workflows. See [`frontend/README.md`](frontend/README.md).
- `model-development/`: data sources, audio/transcript preprocessing notebooks, the 154,828-utterance
  dataset, the SPEAK-ASR baseline, and LoRA experiments. See
  [`model-development/README.md`](model-development/README.md), and
  [`model-development/INTEGRATION_POINTS.md`](model-development/INTEGRATION_POINTS.md) for how it
  connects to the backend and to preprocessing.
- `final-scripts/`: full fine-tuning (`finetune_whisper.py`), evaluation scripts, the run tracker
  (`finetune_tracker.csv`), and [`finetuneGuide.md`](final-scripts/finetuneGuide.md).
- `ErrorAnalysis/`: clustering-based error analysis of the fine-tuned runs
  ([`Analysis.md`](ErrorAnalysis/Analysis.md)) and the transcript normalization approach that led to the
  v5/v6 datasets.
- `asr-lexicon-tools/`: scripts that find and fix inconsistent spellings/ZWJ usage in the
  training transcripts.
- `openWake/`: the English command audio classifier built on openWakeWord embeddings. See
  [`openWake/README.md`](openWake/README.md).
- `docs/`: deployment, testing and voice-enrollment documentation:
  [`DEPLOYMENT.md`](docs/DEPLOYMENT.md), [`EC2_TEST_RUNBOOK.md`](docs/EC2_TEST_RUNBOOK.md),
  [`TESTING_REPORT.md`](docs/TESTING_REPORT.md), [`voice-enrollment.md`](docs/voice-enrollment.md).
  See also [`DEVOPS.md`](DEVOPS.md) for the deployment reference.
- `project-docs/`: the project's formal deliverables, SRS, Software Architecture Document, ERD,
  Gantt chart, and project proposal.
- `test/`, `scripts/`: backend, deployment, failover and load tests, and the scripts that run them.
- `PRESENTATION_README.md`, `PRESENTATION_SCRIPT_YOHAN.md`: fact sheet and spoken script for the
  final presentation.
- `docker-compose*.yml`, `Caddyfile.example`: local PostgreSQL and the production deployment
  (Caddy + backend on AWS EC2, frontend on Vercel).

## ASR results

Full fine-tune of `openai/whisper-small` on the final corpus (OpenSLR-52 plus a collected
set, speaker-disjoint splits):

| Run | Setup | Test WER | Test CER |
|---|---|---|---|
| run1 | v1 data, batch 32 | 17.08% | 3.49% |
| run5 | v4 (speaker-disjoint) | 19.06% | 4.90% |
| run6 | v5 (register/boundary normalization) | 17.15% | 4.62% |
| run10 | v6, 5 epochs | 17.17% | 4.71% |
| **run11** | **v6, 6 epochs** | **16.38%** | **4.43%** |

WER is only comparable within the same test split (v1 vs v4+ differ). Best recipe: LR 3e-5 linear,
effective batch 64, 500 warmup steps, 6 epochs. The model is published as
`Yohan2003/whisper-small-sinhala-run11-v6-e6`, and the data as `Yohan2003/whisper-sl-data`.
Live captioning uses a CTranslate2 (faster-whisper) conversion of the model. Details are in
[`ErrorAnalysis/Analysis.md`](ErrorAnalysis/Analysis.md) and `final-scripts/finetune_tracker.csv`.

Known cost: full fine-tunes lose most English ability (LibriSpeech WER 4.27% → 80.86% for run1).

## Voice command mode

Students can control the app hands-free. A wake word ("zimi") unlocks one command; a command is
recognised by three stages:

1. **Fuzzy text match** on the ASR transcript (always on, inherits ASR errors).
2. **Speaker-embedding match** (Sinhala): Whisper encoder embedding compared with the student's
   enrolled samples (12 commands × 5 clips). Survives ASR errors.
3. **Audio classifier** (English): MLP on openWakeWord embeddings, no enrollment needed.

Destructive commands use higher thresholds, and "submit" only opens a confirmation dialog. See
[`docs/voice-enrollment.md`](docs/voice-enrollment.md) for the full design.

## Status

The backend supports a fake transcriber (canned output, the default for fast local setup) and real
Whisper-based transcribers selected with `TRANSCRIBER_BACKEND`; see
[`backend/README.md`](backend/README.md). The fine-tuned run11 model, live captioning, command mode and
the AWS deployment are in place. Known limitations:

- On the deployed 2-vCPU instance, median speech-to-command latency was 5.6 s for one user and some
  commands timed out ([`docs/TESTING_REPORT.md`](docs/TESTING_REPORT.md)).
- The English command classifier was trained on TTS and tested on one real speaker; there is
  no Sinhala classifier yet.
- Honorific register and rare names remain the main ASR error sources.
- Manual review of flagged rows in the collected data was still in progress.

## Getting started

- Backend: follow [`backend/README.md`](backend/README.md) to configure PostgreSQL, migrate the
  database, run the API, and start the transcription worker.
- Frontend: `cd frontend`, see [`frontend/README.md`](frontend/README.md) for setup, demo
  accounts, and the mock-vs-real API toggle.
- ASR fine-tuning/evaluation: see [`final-scripts/finetuneGuide.md`](final-scripts/finetuneGuide.md) and
  [`model-development/README.md`](model-development/README.md).
- Deployment: see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and [`DEVOPS.md`](DEVOPS.md).
- Presentation material: [`PRESENTATION_README.md`](PRESENTATION_README.md).
