# SinhaSpeech

An accessible platform for Sinhala-speaking students and teachers: record or upload lectures
and spoken answers, get them transcribed, review/edit the transcript, and run speech-based
quizzes.

## Layout

- `backend/` — FastAPI + PostgreSQL API: auth, media upload, transcription job queue/worker,
  transcript CRUD. See [`backend`](backend) (no dedicated README yet — see `app/main.py` and
  `alembic/` for the schema).
- `frontend/` — React app for lectures, spoken quizzes, study notes, transcript review, and
  teacher workflows. See [`frontend/README.md`](frontend/README.md).
- `research/` — Sinhala ASR research: model evaluation, a Flask record/transcribe demo, and
  the SPEAK-ASR baseline this project builds on. See [`research/README.md`](research/README.md).
- `docs/` — SRS, Software Architecture Document, ERD, Gantt chart, and project proposal.
- `docker-compose.yml` — local PostgreSQL service for the backend.

## Status

The backend currently ships a fake transcriber (canned output) rather than a real ASR model —
see [`research/README.md`](research/README.md) for the evaluated Sinhala model
(`SPEAK-ASR/whisper-si-exp-10-medium-all`, 10.85% WER) that's intended to be wired in next.

## Getting started

- Backend: `cd backend`, see `requirements.txt`, `alembic.ini`, and `docker-compose.yml` for
  the Postgres service.
- Frontend: `cd frontend`, see [`frontend/README.md`](frontend/README.md) for setup, demo
  accounts, and the mock-vs-real API toggle.
- ASR research demo/evaluation: see [`research/README.md`](research/README.md).
