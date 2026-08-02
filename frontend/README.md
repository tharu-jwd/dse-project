# SinhaSpeech Frontend

Accessible React frontend for the Phase 2 SinhaSpeech web application. It supports lecture transcription, spoken self-study notes, speech-based quizzes, transcript correction and teacher review workflows.

## Setup

Requirements: Node.js 20.19+ (or 22.12+) and npm.

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

The development server prints its local URL, normally `http://localhost:5173`.

## Demo accounts

Mock mode is enabled by default.

| Role | Email | Password |
|---|---|---|
| Student | `student@sinhaspeech.lk` | `demo123` |
| Teacher | `teacher@sinhaspeech.lk` | `demo123` |

In mock mode, upload a file with `fail` in its filename (for example `lecture-fail.mp3`) to demonstrate the failed-transcription and retry state.

## Scripts

```bash
npm run dev      # Start the Vite development server
npm run lint     # Run oxlint
npm run build    # Create a production build in dist/
npm run preview  # Preview the production build
```

## Routes

| Route | Access | Purpose |
|---|---|---|
| `/login` | Public | Sign in and demo credentials |
| `/dashboard` | Both roles | Role-aware shortcuts and recent transcripts |
| `/lectures/new` | Both roles | Upload and transcribe a recorded lecture |
| `/notes/new` | Student | Record/upload and transcribe a study note |
| `/transcripts` | Both roles | Search and filter the transcript library |
| `/transcripts/:id` | Both roles | Shared transcript editor |
| `/quizzes`, `/quizzes/:id` | Student | Published quizzes and spoken answers |
| `/teacher/quizzes` | Teacher | Draft/published quiz management |
| `/teacher/quizzes/new` | Teacher | Create a quiz |
| `/teacher/quizzes/:id/edit` | Teacher | Edit and publish a quiz |
| `/teacher/submissions` | Teacher | Student submissions |
| `/teacher/submissions/:id` | Teacher | Mark and feedback form |
| `/settings` | Both roles | Accessibility preferences |
| `/help` | Both roles | English and Sinhala quick start |

Protected routes return unauthenticated users to `/login`. Role-restricted routes return the wrong role to `/dashboard`.

## Mock mode and API mode

Copy `.env.example` to `.env` and choose:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCK_API=true
```

- `true`: uses in-memory demo users, Sinhala transcripts, quizzes, submission data and short processing delays. No backend is required.
- `false`: sends authenticated REST requests to `VITE_API_BASE_URL`.

Restart Vite after changing environment variables. API mode sends the stored JWT in the `Authorization: Bearer <token>` header. A `401` clears the local session and returns the user to sign-in. Passwords, tokens and transcript bodies are never logged.

## Expected backend contract

Login returns `{ token, user }`; `user.role` is `STUDENT` or `TEACHER`. A transcription upload uses `multipart/form-data` fields `file`, `title` and `type`, returning `{ jobId, status }`. Job status eventually returns `{ status: "COMPLETED", transcriptId }` or `{ status: "FAILED", message }`.

```text
POST   /auth/login
POST   /auth/logout
POST   /transcriptions
GET    /transcriptions/jobs/:jobId
GET    /transcripts
GET    /transcripts/:id
PATCH  /transcripts/:id
POST   /transcripts/:id/finalize
GET    /transcripts/:id/export?format=txt|docx|pdf
GET    /quizzes
POST   /quizzes
PATCH  /quizzes/:id
POST   /quizzes/:id/publish
POST   /quizzes/:id/submit
GET    /submissions
GET    /submissions/:id
PATCH  /submissions/:id/review
```

The frontend expects transcript segments to retain their IDs, timestamps, confidence and word metadata when patched. Export endpoints should return a file/blob. Submission endpoints can be adapted in `src/api/index.js` if the backend nests them below quizzes.

## Accessibility and browser support

Transcript size, high contrast and confidence preferences are stored in local storage. The app uses semantic controls, visible focus states, keyboard-accessible navigation and dialogs, and text/icon cues in addition to colour. Browser recording requires `MediaRecorder`, a microphone and permission through a secure context (`https` or localhost). Uploading prerecorded audio remains available when recording is unsupported.
