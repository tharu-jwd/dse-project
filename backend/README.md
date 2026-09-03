# SinhaSpeech Backend

FastAPI and PostgreSQL backend for SinhaSpeech. It provides JWT authentication,
media upload and access, asynchronous transcription jobs, and transcript editing
and finalization.

## Requirements

- Python 3.11 or newer
- PostgreSQL 17 (the root Docker Compose file can run it locally)
- FFmpeg when using a real Whisper-based transcriber

Run all backend commands below from the repository root unless the command starts
with `cd backend`.

## Local setup

1. Create the root environment file:

   ```bash
   cp .env.example .env
   ```

   On PowerShell, use `Copy-Item .env.example .env`.

2. Start PostgreSQL:

   ```bash
   docker compose up -d database
   ```

3. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   # PowerShell: .\.venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```

4. Apply database migrations and optionally add demo data:

   ```bash
   cd backend
   alembic upgrade head
   python -m scripts.seed_users
   python -m scripts.seed_transcripts
   ```

5. Start the API from `backend/`:

   ```bash
   uvicorn app.main:app --reload
   ```

   The API is available at `http://localhost:8000`, with interactive OpenAPI
   documentation at `http://localhost:8000/docs`.

6. In a second terminal, start the transcription worker from `backend/`:

   ```bash
   python -m scripts.run_transcription_worker
   ```

   Use `--once` to process at most one queued job and exit. The API only queues
   uploads; a running worker is required to complete them.

## Configuration

Backend settings are loaded from the repository-root `.env`, not from a file in
`backend/`. See [`.env.example`](../.env.example) for all settings and safe local
defaults.

| Variable | Purpose | Default |
|---|---|---|
| `POSTGRES_DB` | PostgreSQL database | Required |
| `POSTGRES_USER` | PostgreSQL user | Required |
| `POSTGRES_PASSWORD` | PostgreSQL password | Required |
| `POSTGRES_HOST` | Database host seen by the API | `127.0.0.1` |
| `POSTGRES_PORT` | Host PostgreSQL port | `5432` |
| `JWT_SECRET_KEY` | Secret used to sign access tokens | Required |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access-token lifetime | `60` |
| `CORS_ORIGINS` | Comma-separated allowed browser origins | `http://localhost:5173` |
| `MEDIA_STORAGE_DIR` | Upload directory, relative to the repo root or absolute | `storage/uploads` |
| `MAX_UPLOAD_SIZE_BYTES` | Maximum uploaded file size | `104857600` |
| `TRANSCRIBER_BACKEND` | `fake`, `whisper`, or `speak_asr` | `fake` |
| `WHISPER_MODEL` | OpenAI Whisper model name for `whisper` | `small` |
| `WHISPER_LANGUAGE` | Language code for `whisper` | `si` |
| `WHISPER_BASE_MODEL` | Base model for `speak_asr` | `openai/whisper-small` |
| `WHISPER_ADAPTER_MODEL` | LoRA adapter for `speak_asr` | `SPEAK-ASR/whisper-si-exp-10` |

Keep `TRANSCRIBER_BACKEND=fake` for a fast local setup without downloading model
weights. Real transcribers are loaded by the worker and may require substantial
download time and memory.

## API overview

| Method and path | Purpose |
|---|---|
| `GET /health` | API health check |
| `GET /health/database` | Database connectivity check |
| `POST /auth/login` | Authenticate and return a bearer token |
| `GET /auth/me` | Return the authenticated user |
| `POST /auth/logout` | End the client session (JWTs are stateless) |
| `POST /transcriptions` | Upload media and queue a transcription |
| `GET /transcriptions/jobs/{job_id}` | Poll transcription status |
| `GET /transcripts` | List accessible transcripts |
| `GET /transcripts/{transcript_id}` | Get one transcript |
| `PATCH /transcripts/{transcript_id}` | Edit a draft transcript |
| `POST /transcripts/{transcript_id}/finalize` | Finalize a transcript |
| `GET /media/{media_id}` | Stream accessible uploaded media |

All routes except health, root, documentation, and login require
`Authorization: Bearer <token>`.

The upload endpoint accepts `multipart/form-data` fields `file`, `title`, and
`type`. Supported types are `LECTURE`, `NOTE`, and `QUIZ_ANSWER`; notes and quiz
answers must use audio rather than video.

## Demo accounts

After running `python -m scripts.seed_users`:

| Role | Email | Password |
|---|---|---|
| Student | `student@sinhaspeech.lk` | `demo123` |
| Teacher | `teacher@sinhaspeech.lk` | `demo123` |

These credentials are for local development only.

## Database migrations

Run migration commands from `backend/` with the virtual environment active:

```bash
alembic upgrade head
alembic current
alembic revision --autogenerate -m "describe the change"
```

## Project structure

```text
backend/
├── alembic/             Database migrations
├── artifacts/           Committed voice-command analysis summaries
├── docs/                Backend feature documentation
├── app/
│   ├── api/             Routes and request dependencies
│   ├── core/            Configuration and JWT/password security
│   ├── db/              SQLAlchemy engine and sessions
│   ├── models/          Database models
│   ├── schemas/         API validation and response models
│   ├── services/        Application and storage logic
│   └── transcribers/    Fake, Whisper, and SPEAK-ASR adapters
└── scripts/             Seeders and transcription worker
```

See [Voice command enrollment](docs/voice-enrollment.md) for enrollment,
embedding validation, runtime matching, and A/B testing.
