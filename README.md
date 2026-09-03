# SinhaSpeech

An accessible platform for Sinhala-speaking students and teachers: record or upload lectures
and spoken answers, get them transcribed, review/edit the transcript, and run speech-based
quizzes.

## Layout

- `backend/`: FastAPI + PostgreSQL API for auth, media upload, transcription job queue/worker,
  and transcript CRUD. See [`backend/README.md`](backend/README.md).
- `frontend/`: React app for lectures, spoken quizzes, study notes, transcript review, and
  teacher workflows. See [`frontend/README.md`](frontend/README.md).
- `model-development/`: Sinhala ASR data preparation, Whisper training,
  evaluation, experiment tracking, and research. See
  [`model-development/README.md`](model-development/README.md).
- `project-docs/`: the project's formal deliverables, SRS, Software Architecture Document, ERD, Gantt
  chart, and project proposal.
- `docker-compose.yml`: local PostgreSQL service for the backend.

## Status

The backend currently ships a fake transcriber (canned output) rather than a real ASR model.
See [`model-development/README.md`](model-development/README.md) for current
fine-tuning work and
[`model-development/docs/integration-points.md`](model-development/docs/integration-points.md)
for the backend swap-in point.

## Getting started

- Backend: follow [`backend/README.md`](backend/README.md) to configure PostgreSQL, migrate the
  database, run the API, and start the transcription worker.
- Frontend: `cd frontend`, see [`frontend/README.md`](frontend/README.md) for setup, demo
  accounts, and the mock-vs-real API toggle.
- ASR fine-tuning/evaluation: see [`model-development/README.md`](model-development/README.md).
