# SinhaSpeech

An accessible platform for Sinhala-speaking students and teachers: record or upload lectures
and spoken answers, get them transcribed, review/edit the transcript, and run speech-based
quizzes.

## Layout

- `backend/`: FastAPI + PostgreSQL API for auth, media upload, transcription job queue/worker,
  and transcript CRUD. See [`backend/README.md`](backend/README.md).
- `frontend/`: React app for lectures, spoken quizzes, study notes, transcript review, and
  teacher workflows. See [`frontend/README.md`](frontend/README.md).
- `model-dev-tharupahan/`: clean-room Sinhala ASR rebuild with explicit data,
  evaluation, reproducibility, and GPU-cost gates. See
  [`model-dev-tharupahan/README.md`](model-dev-tharupahan/README.md).
- `project-docs/`: the project's formal deliverables, SRS, Software Architecture Document, ERD, Gantt
  chart, and project proposal.
- `docker-compose.yml`: local PostgreSQL service for the backend.

## Status

The backend supports a lightweight fake transcriber, complete Whisper
checkpoints, and SPEAK-ASR/PEFT adapters through configuration. See
[`model-dev-tharupahan/README.md`](model-dev-tharupahan/README.md) for current
fine-tuning work and [`backend/README.md`](backend/README.md) for transcriber
configuration.

## Getting started

- Backend: follow [`backend/README.md`](backend/README.md) to configure PostgreSQL, migrate the
  database, run the API, and start the transcription worker.
- Frontend: `cd frontend`, see [`frontend/README.md`](frontend/README.md) for setup, demo
  accounts, and the mock-vs-real API toggle.
- ASR fine-tuning/evaluation: see
  [`model-dev-tharupahan/README.md`](model-dev-tharupahan/README.md).
