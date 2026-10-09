# SinhaSpeech feature list

SinhaSpeech is an accessible platform for Sinhala-speaking students and teachers. It turns Sinhala
speech into clear, editable text for lectures, learning and assessment, and serves Deaf and
hard-of-hearing students (who need the teacher's speech as text) and students with writing
difficulties (who answer by speaking).

Each feature below names the demo video that records it
(`python scripts/demo-videos/record_demos.py`, see `scripts/demo-videos/`).

## For everyone

| Feature | What it does | Demo scene |
|---|---|---|
| **Sinhala speech-to-text** | A full fine-tune of Whisper-small for Sinhala (16.38% WER / 4.43% CER on a speaker-disjoint test set) powers every transcript. | all |
| **Sign-in with roles** | Student and teacher accounts with JWT authentication. The UI and routes differ per role, and role-restricted pages redirect the wrong role to the dashboard. Demo accounts can be filled in with one click. | `login` |
| **Role-aware dashboard** | Shortcuts to the tools each role needs, recent transcripts and stats. | `dashboard` |
| **Lecture captioning (upload)** | Upload a recorded lecture (audio or video). The app uploads, queues and transcribes it, shows progress and failure/retry states, then opens the transcript. | `lecture_upload` |
| **Transcript editor** | Review and correct a transcript segment by segment. Low-confidence words are highlighted, the threshold is adjustable, and smart search and replace fixes a word everywhere. Shows word count, duration and reading time, with synced audio playback. Save drafts, then finalize behind a confirmation step. | `transcript_editor` |
| **Export** | Download transcripts as TXT, DOCX or PDF (PDF for lectures). | `transcript_editor`, `transcript_library` |
| **Transcript library** | One searchable place for lectures, study notes and quiz answers, with filters by type and status, open, export and delete. | `transcript_library` |
| **Accessibility settings** | English or Sinhala interface, three transcript text sizes, high-contrast mode, and a choice of keyboard-and-mouse or voice-command interaction. Settings are saved on the device. | `accessibility_settings` |
| **Quick start and help** | Bilingual (English and Sinhala) guide inside the app. | `help` |
| **Accessibility by design** | Keyboard-only sign-in, skip-to-content link, ARIA labelling, and axe (WCAG A/AA) checks in the end-to-end tests. | `login`, `accessibility_settings` |

## For students

| Feature | What it does | Demo scene |
|---|---|---|
| **Self-study notes** | Record or upload a voice note and get an editable transcript, or switch to live transcription. | `self_study_notes` |
| **Live captioning** | Streaming transcription over WebSocket: words appear as the speaker talks, with pause and resume, voice-activity indication, and click-to-edit text that autosaves. Uses a CTranslate2 (`faster-whisper`) conversion of the model. | `live_captioning` (backend) |
| **Spoken quizzes** | Answer quiz questions by speaking, either by recording a clip or with live transcription, and review the transcript before submitting. Spoken and multiple-choice questions are supported. | `spoken_quiz` |

## For teachers

| Feature | What it does | Demo scene |
|---|---|---|
| **Quiz management** | Create, edit, reorder and publish speech-based quizzes. Questions are spoken or multiple choice, with draft and published states. | `teacher_quizzes` |
| **Submission review** | Read each student's answer with its transcript and give marks and feedback. | `teacher_review` |
| **Lecture captions** | Upload recorded lectures and caption them for students. | `lecture_upload` (teacher also has this) |

## Hands-free voice commands

| Feature | What it does | Demo scene |
|---|---|---|
| **Wake word + command** | Say the wake word ("zimi") to unlock one command. Commands include next, previous, stop, save, submit, delete, cancel, answer, and options one to four. Available in Sinhala and English. | `voice_commands_live` (backend) |
| **Three-stage recognition** | (1) Fuzzy text match on the transcript, (2) speaker-embedding match against the student's own enrolled samples, (3) an English audio classifier on openWakeWord embeddings. | `voice_command_setup` (backend) |
| **Safety for destructive commands** | Submit and delete use higher thresholds, and "submit" only opens a confirmation dialog. | `voice_commands_live` (backend) |
| **Personal enrollment** | Students record several samples per command so the app recognizes how they say it. | `voice_command_setup` (backend) |

## Platform and ML

| Feature | What it does |
|---|---|
| **Fine-tuned ASR** | Whisper-small fully fine-tuned on ~154,828 Sinhala utterances (OpenSLR-52 plus collected data) across six versioned splits. Models and data are published in the SinhaSpeech Hugging Face organization. |
| **Error analysis** | Clustering-based analysis of fine-tuned runs drove data normalization (spacing, register, spelling), producing the v4 to v6 datasets. |
| **Pluggable transcribers** | Backend supports a fake transcriber (canned output for fast local setup), a Whisper transcriber and a SPEAK-ASR baseline, chosen with `TRANSCRIBER_BACKEND`. |
| **Async transcription pipeline** | FastAPI + PostgreSQL API, a job queue and worker, windowing for recordings longer than 30 s, and ffmpeg decoding of many media formats. |
| **Mock mode** | The frontend can run against an in-memory demo API with no backend, which the demo-video script uses. |
| **Deployment** | Docker Compose, Caddy and the backend on AWS EC2, the frontend on Vercel. |

## Known limitations (be upfront in marketing)

- On the 2-vCPU EC2 instance, median speech-to-command latency was 5.6 s for one user and some
  commands timed out.
- The English command classifier was trained on TTS and tested on one real speaker. There is no
  Sinhala classifier yet.
- Honorific register and rare names remain the main ASR error sources.
- The model has not been evaluated for English or mixed-language audio. An earlier full
  fine-tune lost most English ability.
