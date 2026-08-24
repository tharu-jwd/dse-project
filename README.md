# SinhaSpeech

An accessible platform for Sinhala-speaking students and teachers: record or upload lectures
and spoken answers, get them transcribed, review/edit the transcript, and run speech-based
quizzes.

## Layout

- `backend/`: FastAPI + PostgreSQL API for auth, media upload, transcription job queue/worker,
  and transcript CRUD. See [`backend/README.md`](backend/README.md).
- `frontend/`: React app for lectures, spoken quizzes, study notes, transcript review, and
  teacher workflows. See [`frontend/README.md`](frontend/README.md).
- `model-development/`: Sinhala ASR, LoRA fine-tuning on Whisper, evaluation by WER, and the
  SPEAK-ASR baseline this project builds on. See
  [`model-development/README.md`](model-development/README.md), and
  [`model-development/INTEGRATION_POINTS.md`](model-development/INTEGRATION_POINTS.md) for how
  it connects to the backend and to preprocessing.
- `project-docs/`: the project's formal deliverables, SRS, Software Architecture Document, ERD, Gantt
  chart, and project proposal.
- `docker-compose.yml`: local PostgreSQL service for the backend, and a shared MLflow tracking
  server for fine-tuning experiments, opt-in via `--use-mlflow` (see
  [`final-scripts/finetuneGuide.md`](final-scripts/finetuneGuide.md#mlflow-experiment-tracking-optional)).

## Status

The backend currently ships a fake transcriber (canned output) rather than a real ASR model.
See [`model-development/README.md`](model-development/README.md) for the evaluated Sinhala
baseline (`SPEAK-ASR/whisper-si-exp-10-medium-all`, 10.85% WER) and the in-progress fine-tuning
work intended to replace it, and
[`model-development/INTEGRATION_POINTS.md`](model-development/INTEGRATION_POINTS.md) for exactly
where that swap-in point is.

## Getting started

- Backend: follow [`backend/README.md`](backend/README.md) to configure PostgreSQL, migrate the
  database, run the API, and start the transcription worker.
- Frontend: `cd frontend`, see [`frontend/README.md`](frontend/README.md) for setup, demo
  accounts, and the mock-vs-real API toggle.
- ASR fine-tuning/evaluation: see [`model-development/README.md`](model-development/README.md).
