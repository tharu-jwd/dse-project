# SinhaSpeech — Master Test Plan

Last updated 2026-09-20. Structured against the Master Test Plan template (Rational Unified
Process format). Every "Status" line below reflects what actually exists and passes as of
this date — not what is planned — with plans separated out explicitly in §3.1's tables.

**At a glance:**

| Technique (§3.1) | Status |
|---|---|
| 3.1.1 Data & Database Integrity | 🟢 Strong (one narrow gap) |
| 3.1.2 Function Testing | 🟢 Complete for the current API |
| 3.1.3 User Interface Testing | 🟢 Implemented (component + e2e + axe; see Appendix A) |
| 3.1.4 Performance Profiling | 🟡 Partial (N+1 fixed; model-dependent latency still manual) |
| 3.1.5 Load Testing | 🟢 Measured live on production 2026-09-20: crash fix confirmed holding; inference latency itself is borderline even at 1 user |
| 3.1.6 Security & Access Control | 🟢 Implemented |
| 3.1.7 Failover & Recovery | 🟢 Implemented + a real production reboot verified 2026-09-20 (§3.1.7) |
| 3.1.8 Configuration Testing | 🟢 Server side plus a 4-engine browser matrix (found a Safari defect) |

```
378 backend tests   (pytest, test/backend/ — 305 confirmed 2026-09-20 against production's
                      disposable database, +73 added later the same day: voice-enrolment/
                      voice-sample/media API tests, stuck-job recovery, destructive-command
                      safety, clean-install/compose validation, streaming persistence,
                      WebSocket auth dependency. Statement coverage 79% → 84.16%, gated in
                      CI at 75%. Run via `scripts/test-isolated.sh` for a fully isolated,
                      no-.env, empty-database run, or `scripts/test-plan.sh` for the full
                      one-command backend+frontend+e2e+smoke sweep with evidence in reports/.)
 11 deployment smoke tests   (pytest, test/deployment/ — run against the live system,
                                including immediately after the 2026-09-20 reboot test)
 43 frontend component tests, 72 Playwright e2e tests   (both green as of 2026-09-20; the +12
                                                            over the earlier 60 are
                                                            immersion.spec.js's leaked-internals
                                                            check (8) and quiz-and-editor-a11y.spec.js
                                                            (4, one data-state skip) — see
                                                            reports/2026-09-20/summary.md)
```

---

## 1. Evaluation Mission and Test Motivation

SinhaSpeech is a Sinhala speech-to-text and voice-command web application built for students
who find typing or using a mouse difficult. Its two properties that most shape the test
approach:

- **It is speech infrastructure, not a CRUD app.** A Whisper model runs on both a live
  WebSocket path and a background batch path, real audio drives fuzzy-text and voice-fingerprint
  matching, and the correctness bar is "does the model behave sensibly on messy real speech,"
  not just "does the function return the right value for a fixed input."
- **Accessibility is the actual requirement**, not a nice-to-have layered on afterward. A bug
  that makes the app merely inconvenient for a typical user can make it *unusable* for the
  student it was built for.

The mission for this test effort is to:

- Find defects in the matching, streaming, and voice-command logic before they reach a
  student in production — this is where the highest-consequence bugs live, since a false
  match on a destructive command (delete, submit) directly destroys a student's work.
- Verify that role-based access control actually holds under a real signed token, not just
  that the guard code exists.
- Establish what is *not yet known* about the system (load behaviour, failover behaviour,
  cross-browser behaviour) so those are visible gaps rather than silent assumptions.
- Keep every finding traceable to a specific failing input or condition, since "the model is
  sometimes wrong" is not actionable and "eka scores 83% against deka because of a shared
  prefix" is.

---

## 2. Target Test Items

| Item | What it is | Primarily tested by |
|---|---|---|
| **Backend API** (FastAPI, 27 REST endpoints + 1 WebSocket) | Auth, transcripts, quizzes, submissions, voice enrollment, streaming | §3.1.2, §3.1.6 |
| **Matching logic** (`app/streaming/commands.py`, `command_resolution.py`, `embeddings.py`) | Fuzzy-text and voice-fingerprint command matching, the wake-word gate | §3.1.2 (heaviest coverage — see §5 for why) |
| **Data models** (Postgres 17, 13 tables, Alembic-migrated) | Users, transcripts, quizzes, submissions, voice enrollments, the job queue | §3.1.1 |
| **Background worker** | Polls the job queue, runs batch transcription | §3.1.2 (the queue → claim → transcript pipeline), §3.1.7 (failover, untested) |
| **Frontend** (React 19 / Vite) | The UI students and teachers actually use | §3.1.3 (untested) |
| **Deployment** (AWS EC2 + Docker Compose + Caddy, Vercel) | The live system end-to-end | §2 deployment smoke tests |
| **CI** (GitHub Actions) | Regression gate on every push/PR | §4 |

---

## 3. Test Approach

### 3.1 Testing Techniques and Types

#### 3.1.1 Data and Database Integrity Testing

**Status: 🟢 Strong — constraints, cascades, queue concurrency and transaction atomicity are all covered; one narrow gap remains.**

| Technique Objective | Exercise the database and its access methods independent of the UI — schema creation, foreign keys, and per-row correctness — to observe data corruption or incorrect persistence. |
|---|---|
| **Technique** | `test_voice_enrollment.py` (18 tests) inserts and deletes real rows against a live Postgres instance: samples accepted/rejected by similarity, per-language isolation, cascade delete on user removal, unknown-id rejection. `test_db_integrity.py` (5 tests) exercises constraint and foreign-key behaviour directly against Postgres, independent of the service layer: `uq_command_enrollments_slot` rejects a duplicate `(user, command, language, sample_index)` row while correctly allowing the same `sample_index` under a different language; deleting a user cascades their voice enrollments (`ON DELETE CASCADE`); deleting a teacher who owns quizzes is *rejected* by the database (`ON DELETE RESTRICT` on `quizzes.created_by`); deleting a quiz cascades its questions (`ON DELETE CASCADE` on `questions.quiz_id`). In CI, `alembic upgrade head` runs against a disposable Postgres 17 service container on every push, so schema creation itself is verified on every commit, not just once. |
| **Oracles** | Direct row inspection via SQLAlchemy queries in the test body; `pytest.raises(IntegrityError)` for constraint violations. |
| **Required Tools** | pytest, SQLAlchemy, Postgres 17 (Docker for CI, dev instance for local runs) |
| **Job-queue concurrency and atomicity** | `test_db_concurrency.py` (6 tests). **Concurrency:** `claim_next_job()` relies on `SELECT … FOR UPDATE SKIP LOCKED` and had only ever run with one worker. Verified deterministically with two live database sessions — while worker A holds a lock on the oldest job, worker B is handed the *other* job (never the locked one, never a wait); with a single locked job B gets nothing back within a timeout instead of blocking; and the real `claim_next_job()` called from two threads over four jobs never returns the same job twice and never leaves one unclaimed. A mutation check confirmed the blocking test genuinely detects the failure: with `skip_locked` removed, the second worker blocks. **Atomicity and failure handling:** a transcriber that raises marks the job `FAILED` with a generic message (an internal exception string was confirmed not to leak into the user-visible error); a result the database rejects *inside* `complete_job()`'s transaction (built with `model_construct` to bypass Pydantic and reach the `ck_transcript_segments_time_range` CHECK constraint) leaves the job `FAILED` with no partial `Transcript` row behind; a media file that vanished from disk fails the job with its own message. |
| **Success Criteria** | All 29 tests pass; migrations apply cleanly to an empty database. |
| **Note on running these locally** | Because they write to the shared dev Postgres instance, cleanup is intentional and verified rather than assumed: fixture teardown (`conftest.py`'s `_delete_user`) explicitly removes quiz submissions, quizzes, transcripts, transcript segments, transcription jobs, and media files owned by a fixture user, in an order that respects each table's real `ondelete` policy, before deleting the user row itself. Confirmed by running the full suite three times in a row (including once under CI's exact environment) and checking the database directly each time for leftover fixture rows or stuck job-queue entries — none found. |
| **Special Considerations** | Locally, these tests still run against the shared dev database rather than an isolated one (see §5, risk 1) — CI is the hermetic check. Writing the RESTRICT/CASCADE tests required checking each foreign key's actual `ondelete` setting in the model file rather than assuming — the two policies are deliberately different (`RESTRICT` protects a teacher's quiz history from silently vanishing; `CASCADE` on child rows like questions/options is correct because they are meaningless without their parent). |

**Not yet done:**
- Concurrent writes to the *same transcript* (two simultaneous edits) — the queue's
  concurrency is covered, but there is no optimistic-locking or last-write-wins behaviour
  defined for transcript edits, so there is not yet an intended behaviour to assert

#### 3.1.2 Function Testing

**Status: 🟢 Complete for the current API — matching logic, quiz lifecycle (MCQ and spoken), transcripts, upload pipeline, systematic input validation, voice-enrolment/voice-sample/media routes, destructive-command safety, and stuck-job recovery (all added 2026-09-20, `TEST_GAPS_README.md`).**

| Technique Objective | Exercise target functionality — navigation, data entry, processing, retrieval — via black-box interaction, verifying business rules are correctly applied for both valid and invalid input. |
|---|---|
| **Technique** | Five layers. (1) **Matching/streaming logic** — 102 tests (`test_commands.py`, `test_command_resolution.py`, `test_embeddings.py`, `test_streaming_buffer.py`, `test_streaming_commands_route.py`, `test_inference_kwargs.py`) drive the fuzzy-text matcher, the voice-embedding matcher, their combination rule, the rolling audio buffer, and the full wake-word gate end-to-end with a fake WebSocket — no real model or database. (2) **Quiz lifecycle** — 16 tests (`test_api_quiz_lifecycle.py`) drive the real FastAPI app via `TestClient`: a teacher creates a quiz, it is correctly hidden from students until published, a student answer is validated against the actual question/option it belongs to, a student's correct-answer view never leaks `isCorrect`, resubmitting updates rather than duplicates a submission, a teacher marks it, and a second teacher is confirmed unable to review a submission for a quiz they don't own. **Spoken answers** (which carry no text — they reference a transcript the student already owns) are covered too: accepted for the student's own transcript, rejected (400) for another student's, for a missing `transcriptId`, and for one that does not exist; an MCQ answer without a selected option is rejected. (3) **Transcripts** — 17 tests (`test_api_transcripts.py`) cover read/update/export/delete/finalize: an owner can edit and export; a second student cannot read, edit, export, or delete another student's transcript; a teacher can read *any* LECTURE transcript by design but not a student's private NOTE, and critically cannot edit a lecture they can only read (the read carve-out does not imply write access); a finalized transcript rejects further edits; renaming to an already-used title is rejected. (4) **Upload → worker pipeline** — 7 tests (`test_api_upload_pipeline.py`) drive `POST /transcriptions` through `TestClient` exactly as a browser would, then hand the queued job to the same `process_next_job()` function the real worker process loops on (using `FakeTranscriber`, the project's own fake backend for exactly this purpose) — covering upload validation (unsupported type, empty file, a note requiring audio not video), job-status privacy, and the full queue → claim → transcript-appears path end to end. (5) **API surface / auth** — 29 tests (`test_api_access_control.py`), see §3.1.6. (6) **Systematic input validation** — 50 tests (`test_api_validation.py`): every JSON write endpoint (login, quiz create/update/submit, submission review, transcript update, voice-enrollment language) is sent twelve deliberately malformed bodies — wrong types, missing fields, 5,000-character strings, an SQL-injection-shaped title, nesting where a scalar belongs — and the property asserted is that **none produces a 5xx** (a 500 is an unhandled exception). Seven read/write routes are also sent malformed path ids (`not-a-uuid`, `%00`, `../../etc/passwd`, a 300-character string) and must answer with a 4xx. Specific rules are pinned individually: login requires both fields and a well-formed email; a quiz title is required and capped at 255 characters; MCQ needs exactly one correct option; a review mark outside 0–100 is rejected; an upload needs a file and a title and a known type. No malformed input in the sweep produced a server error. |
| **Oracles** | Direct assertion against expected return values (`match_command()`'s score, `resolve_command()`'s outcome) and HTTP status codes / response bodies for the API layer. |
| **Required Tools** | pytest, pytest-asyncio, FastAPI `TestClient`, PyJWT |
| **Success Criteria** | All identified use-case flows for the covered areas pass with both valid and invalid input. |
| **Special Considerations** | The matching-logic tests are unusually thorough for a student project because a real bug was found and fixed here mid-project (see §5's finding write-up in the git history) — every edge case in that table is a real failure mode that was hit, not a hypothetical. Writing the quiz lifecycle tests surfaced an undocumented validation rule (`QuizCreate` requires MCQ questions to have exactly 4 options) that was not obvious from the route code alone — found by running the test and reading the resulting 422, not by reading the schema first. The upload pipeline tests run against the shared dev database and `process_next_job()` always claims the *oldest* queued job across the whole table — a naive test could accidentally claim and "complete" a real, unrelated in-flight job with `FakeTranscriber`'s canned text. Guarded against by checking the real queue is empty before those tests run and skipping (not forcing) otherwise — see `_require_empty_queue()`. Two more traps found writing the newest batch: `create_transcription_job()` silently cancels an earlier still-pending job for the same user+title, so reusing a title across two uploads in one test deletes the first job without any error (documented in `test_stuck_job_recovery.py`, not fixed — it's intentional dedup behaviour, but worth a UX check); and `resolve_stored_media()` itself raises `FileNotFoundError` for a missing file rather than only failing on `.unlink()`, so cleanup helpers that delete a file mid-test must catch it explicitly. |
| **(7) Voice-enrolment, voice-sample, and media-access routes** | 32 tests (`test_api_voice_enrollment.py`, `test_api_media.py`) close a real coverage gap: the matching/storage logic underneath these routes was already fully tested (`test_voice_enrollment.py`), but nothing had ever driven the routes themselves through `TestClient` — auth requirement, request validation, and permission checks were untested at the HTTP layer. Covers every endpoint's happy path, missing-auth (401), unknown-command-id (404), empty-upload (400), and — for media — ownership scoping (owner can fetch, a different student cannot, a teacher cannot read a student's private NOTE but can read media behind their own LECTURE transcript). The embedding call (`StreamingTranscriber.embed`) is monkeypatched, same pattern as `test_streaming_commands_route.py`, so no model loads; ffmpeg conversion runs for real. The `/voice-samples` dev-data-collection routes write straight to `settings.voice_samples_dir`, which in dev points at this repo's real training-data directory — every test here isolates that to a `tmp_path` first, since a naive test's `DELETE` call would otherwise remove real recorded samples. |
| **(8) Stuck-job recovery** | 3 tests (`test_stuck_job_recovery.py`) close a previously undefined behaviour: a transcription job interrupted while `PROCESSING` (worker crashed or was killed mid-job) had no code path that ever revisited it — `claim_next_job()` only ever looked at `QUEUED` rows. Fixed with a new `transcription_stuck_job_timeout_minutes` setting (default 30): `claim_next_job()` now also reclaims a `PROCESSING` job whose `started_at` is older than the timeout. Verified: a stale `PROCESSING` job is reclaimed; a fresh one is correctly left alone (so two workers can never run the same job at once); a job whose media file was deleted fails cleanly with a message and does not block the next queued job. |
| **(9) Destructive-command safety** | 28 tests (`test_command_safety.py`) — a deliberately adversarial suite, not an ordinary matching test: 21 near-miss phrases (Sinhala and English) against `submit`/`delete` — truncations, wrong verb tense, shared prefixes, near-neighbour English words — asserting **none** resolves to a destructive command, plus wake-word-mandatory checks (`split_wake_prefix`) and a confirmation that the real phrases still work. **Found a real defect**: see Appendix A.2 finding 6. |

**Not yet done:**
- Export formats other than `txt` (the route only supports `txt` and rejects the rest, which
  is tested; docx/pdf export was in the frontend's contract but is not implemented server-side)
- The WebSocket streaming route is covered with a fake socket (§3.1.2 layer 1) but has no
  test through a real WebSocket handshake against the running app — the deployment smoke
  tests (§4.1) cover that path only for connect/auth, not for a full audio session

#### 3.1.3 User Interface Testing

**Status: 🟢 Implemented.** 23 component tests (in CI) and 8 browser end-to-end tests. Found a
WCAG AA contrast failure on the login button, now fixed. See Appendix A.

| Technique Objective | Exercise navigation, form submission, and object states to observe standards conformance and target behavior; for this application specifically, exercise keyboard-only and screen-reader interaction paths since accessibility is the core requirement. |
|---|---|
| **Technique** | Vitest + React Testing Library for component-level tests, covering `useVoiceCommands.js` (the hook where the real production bug in §5 lived on the client side), `AccessibilityControls` and `VoiceMeter`. Playwright drives real browser flows against the mock API: login, keyboard-only sign-in, protected-route redirects and role boundaries. The quiz/MCQ answer flow and the transcript editor are not yet covered. |
| **Oracles** | Rendered DOM assertions; for accessibility, `axe-core` or equivalent for WCAG contrast/label checks; manual verification of the high-contrast and text-size settings. |
| **Required Tools** | Vitest, React Testing Library, Playwright, axe-core (none installed yet) |
| **Success Criteria** | Not yet defined — pending tool installation. |
| **Special Considerations** | `frontend/package.json` currently has no test script at all — `dev`, `build`, `lint` (oxlint), `preview` only. This is the most conspicuous gap for an application whose entire purpose is usability. |

#### 3.1.4 Performance Profiling

**Status: 🟡 Partial — API latency, query counts, matching speed and buffer memory are now automated checks; anything that needs a loaded Whisper model is still manual.**

| Technique Objective | Measure response times, transaction rates, and resource usage under normal anticipated workload to verify performance requirements. |
|---|---|
| **Technique** | `backend/scripts/benchmark_command_latency.py` measures voice-command round-trip latency. Model quality is measured directly: WER/CER across 6 fine-tuning runs (see §3a in the training-side error analysis, `ErrorAnalysis/`). Container memory was measured live on the deployed server (backend ~1.2 GB, worker ~330 MB idle) during deployment. Frontend payload was measured and reduced from ~46 MB to ~1 MB of images. **Automated (`test_performance.py`, 10 tests):** in-process response-time budgets for four endpoints (`/health`, `/auth/me`, `/transcripts`, `/quizzes`; measured p95 of 2, 8, 10 and 26 ms locally against budgets of 250 ms p95 / 100 ms median — an order of magnitude of headroom, so a failure means something got dramatically slower rather than that a CI runner was busy); SQL-statement counting via a SQLAlchemy event listener to detect N+1 queries; latency of fuzzy matching, embedding matching over a realistic 65-vector bank (13 commands × 5 samples × 768 dims), and full command resolution; and a ten-minute simulated continuous-speech session confirming the streaming buffer stays capped at 15 s with under 20 MB of traced memory and no audio lost or double-counted across repeated force-cuts. |
| **Oracles** | Timed samples compared against fixed budgets (p50/p95); SQL statement counts compared across data sizes; `tracemalloc` peak. Model quality (WER/CER) is still compared across runs by inspection. |
| **Required Tools** | The benchmark script, `docker stats`, browser DevTools network panel |
| **Success Criteria** | The four budgets above hold; the teacher quiz list issues a constant number of SQL statements regardless of quiz count; the streaming buffer never exceeds its cap. |
| **Special Considerations** | **A real defect was found by the query-count test:** `GET /quizzes` as a *student* issues one extra SQL statement per published quiz — measured 9, 13, 18 and 28 statements for 1, 5, 10 and 20 quizzes (exactly 8 + N) — because `serialize_quiz_student()` looks up the student's own submission inside the list comprehension. The teacher's listing, which eager-loads, stays constant at 4. It is harmless at classroom scale against a local database but every extra statement is a network round trip to a managed database. It is recorded as a `strict` expected failure (`xfail(strict=True)`), so the suite stays green, the defect is documented in the test itself, and the marker will start failing the moment the query is fixed, forcing its removal. These measurements are **in-process**: they exclude network latency to the deployed server, which is a deployment property (us-east-1) rather than a code property. |

**Not yet done — all of it depends on a loaded Whisper model, which is why it is not in the automated suite:**
- Time-to-first-caption during live streaming
- Voice-command end-to-end latency (speak → action) as an automated benchmark rather than a
  one-off script run (`benchmark_command_latency.py` exists and is run by hand)
- Process memory growth of the *whole backend* over a long real session (the buffer's own
  growth is covered above; the model's is not)
- ~~Fixing the student quiz-list N+1 described above~~ — **done.** `_own_submissions_by_quiz()`
  now loads every listed quiz's submission in one `IN` query. Statement count went from 8 + N
  (9, 13, 18, 28 for 1, 5, 10, 20 quizzes) to a flat 5 regardless of quiz count. The
  `xfail(strict=True)` marker failed the moment the fix landed, as designed, and was removed.

#### 3.1.5 Load Testing

**Status: 🟢 Measured, including a direct run against the live production instance
(2026-09-20).** Two separate findings, and they must not be conflated:

1. **Crash resistance — fixed and confirmed.** Running the harness against production
   *before* deploying the VAD lock fix (§3.1.4, Appendix A.2 finding 1) reproduced the crash
   directly: the backend container restarted 4 times under load as light as 1-2 concurrent
   streaming users, producing 502s from Caddy and cascading connection failures. After
   deploying the fix, the identical sweep (1, 2, 3, 5, 8 users, 3 minutes each) produced
   **zero container restarts and zero errors in the backend log**, confirmed via
   `docker inspect --format '{{.RestartCount}}'` before and after.
2. **Inference latency — real and unresolved.** Independent of the crash, and visible even
   in the cleanest tier: at **1** concurrent streaming user, median "speech → verdict"
   latency was **5.6s**, and **37.5% of commands (3 of 8) hit the 30-second timeout with no
   verdict at all.** This is on the actual deployed hardware, saying "zimi <command>" for
   real, not a scratch-stack estimate. Latency did not clearly improve or worsen with
   concurrency (median 5.6s → 6.6s → 8.1s → 9.1s → 8.4s across 1/2/3/5/8 users) — the failure
   rate is dominated by per-request variance, not a clean concurrency cliff. **5+ users is
   confounded** by the per-account streaming session cap (`streaming_max_sessions_per_user`,
   default 3): since all simulated users share one demo account, most "failures" at those
   tiers are `"Too many concurrent streaming sessions"` rejections, not real capacity
   failures — this is a known limitation of testing against a single seeded account, noted in
   `locustfile.py` itself.

**The practical conclusion:** the server no longer falls over under concurrent streaming
load, but a single real user already has worse-than-usable odds of a timed-out command on
this 2-vCPU instance. The earlier "1-2 concurrent users" ceiling, measured on a scratch
stack, undersold the problem — the real bottleneck is baseline inference speed on this
hardware, not the number of simultaneous users.

| Technique Objective | Subject the system to varying workloads — normal, worst-case, and concurrent-user — to determine whether it continues to function correctly beyond expected maximum load. |
|---|---|
| **Technique** | Locust (`test/load/locustfile.py`) drives the REST endpoints and concurrent WebSocket streaming sessions, feeding real 16 kHz clips at true real-time pace to find the point where transcription latency becomes unusable. First run against the scratch stack (Appendix A.4), capped at 2 CPUs to resemble the deployed instance; later run directly against the live production instance (`docs/EC2_TEST_RUNBOOK.md` Part 2) using real recorded "zimi <command>" clips, both before and after deploying the VAD fix. Concurrent file uploads / worker queue depth are not yet scripted. |
| **Oracles** | Response-time and error-rate thresholds under load; container restart count and backend log errors as the crash-resistance oracle; the point of failure becomes the finding itself where no formal threshold is set yet. |
| **Required Tools** | Locust or k6 (neither installed yet) |
| **Success Criteria** | Not yet defined. |
| **Special Considerations** | The production deployment runs CPU-only Whisper inference on 2 vCPUs. Every concurrent streaming session competes for the same CPU budget, and there is currently no data on how many simultaneous users the system tolerates before degrading — this is an unknown, not a measured limit, and is the most likely source of a live-demo failure under real classroom load. |

#### 3.1.6 Security and Access Control Testing

**Status: 🟢 Implemented.**

Covers both levels the template distinguishes: application-level (an actor reaches only the
functions and data their role permits) and system-level (only authenticated callers get in at
all, through legitimate tokens).

| Technique Objective | Verify that an actor can access only those functions or data for which their user type has permission, and that only actors with a valid, current, correctly-signed session can reach the system at all. |
|---|---|
| **Technique** | 29 tests in `test_api_access_control.py`, run against the real FastAPI app via `TestClient` with genuine signed JWTs — the actual authentication dependency executes, nothing is mocked out. **System-level:** 6 protected endpoints checked for rejecting no token, a garbage token, an expired token, a token signed with the wrong secret, and a token for a since-deleted user; a valid token is separately confirmed to work, so the negative cases are meaningful; login is confirmed not to distinguish "unknown account" from "wrong password" in its response (no account enumeration). **Application-level:** a student cannot create a quiz, cannot submit as if reviewing, cannot list or read another student's submissions or transcripts; a teacher cannot submit quiz answers; a teacher's submission list is confirmed scoped to quizzes they created, not all quizzes. **Input validation:** a malformed request body returns 422 rather than an unhandled 500; a path-traversal attempt on the enrollment route is rejected; a nonexistent media id is not served. |
| **Oracles** | HTTP status code (401/403/404/422 as appropriate) and, for the enumeration check, byte-identical response bodies between the two failure cases. |
| **Required Tools** | pytest, FastAPI `TestClient`, PyJWT (to forge an intentionally invalid signature for the negative test) |
| **Success Criteria** | All 29 tests pass; every protected route rejects every invalid-credential case tested. |
| **Special Considerations** | Writing these tests surfaced and fixed a reserved-domain bug in the shared test fixtures (`@example.invalid` emails failed Pydantic's `EmailStr` validation) that would have caused confusing failures in any future test reusing those fixtures — fixed once, centrally, in `conftest.py`. |

**Not yet done:**
- SQL-injection–style fuzzing of query parameters (the ORM makes this lower-risk, but
  unverified)
- Rate limiting / brute-force protection on `/auth/login` (none currently implemented in the
  app, so there is nothing yet to test)
- CSRF is not applicable (token-based auth, no cookies), noted here rather than left silently
  unconsidered

#### 3.1.7 Failover and Recovery Testing

**Status: 🟢 Implemented.** 6 tests in `test/failover/`, run against the disposable scratch
stack. They confirmed the backup restores, `pool_pre_ping` recovers without a restart, and
`restart: unless-stopped` revives a crashed backend — and exposed the dropped-client defect in
Appendix A.2.

| Technique Objective | Simulate failure conditions — power/communication interruption, database failure, incomplete transactions — and verify the system recovers to a known, correct state without data loss. |
|---|---|
| **Technique** | Implemented in `test/failover/test_failover.py`: restore a `pg_dump` backup into a scratch database and verify row-for-row integrity (a backup that has never been restored is a hypothesis, not a backup); kill the database container mid-request and confirm the API recovers via SQLAlchemy's `pool_pre_ping` (now verified); drop a WebSocket connection abruptly mid-session and confirm the server cleans up without an unhandled error and without leaking a session slot; kill the backend process and confirm it returns by itself (`restart: unless-stopped`, now verified). **A real EC2 instance reboot was run against the live production host on 2026-09-20** (`docs/EC2_TEST_RUNBOOK.md` Part 3): backed up the production database first (`pg_dump`, copied off-instance), then `sudo reboot`. SSH access returned in ~2 min, the API health check in a further ~1.5 min (~3.5 min total downtime), all four containers (`database`, `backend`, `worker`, `caddy`) came back automatically with no manual intervention, and all 11 deployment smoke tests (`test/deployment/test_smoke.py`) passed afterward, including a real login, an authenticated query, and both WebSocket tests — confirming data survived and the app genuinely works, not just that it answers a ping. |
| **Oracles** | Row-count and content comparison pre/post restore; HTTP success after a simulated database restart; container status and smoke-test pass/fail after a real instance reboot. |
| **Required Tools** | `pg_dump`/`psql`, Docker, SSH access to the deployment host |
| **Success Criteria** | All four containers return to `Up`/`healthy` without manual intervention; `/health` and `/health/database` respond correctly; `test/deployment/test_smoke.py` passes in full post-recovery. All held on the 2026-09-20 run. |
| **Special Considerations** | This category is inherently somewhat destructive (killing a live container, restarting the instance) — run against a disposable environment where possible, and against the live deployment only with a fresh backup taken first and real user impact accepted (the 2026-09-20 run caused ~3.5 min of real downtime). `docker.service` was confirmed `enabled` at boot, which is what made the automatic recovery possible — had it been disabled, `restart: unless-stopped` alone would not have been enough. |

#### 3.1.8 Configuration Testing

**Status: 🟢 Implemented.** The server-side surface is covered by `test_configuration.py`, and
the browser matrix now runs across Chromium, Firefox, WebKit and a phone viewport — which
immediately found a second configuration defect, this time in Safari's engine (Appendix A.2).
Real devices and network throttling remain untested.

| Technique Objective | Verify correct operation across the different hardware, software, browser, and network configurations the deployed system will actually be used under. |
|---|---|
| **Technique** | Manual testing in Firefox on Linux during deployment surfaced a real, user-facing defect: `sinhaspeech.duckdns.org` is blocked outright by uBlock Origin (and, by the same mechanism, likely AdGuard, Brave Shields, and Pi-hole), because free dynamic-DNS domains are commonly abused by malware and appear on ad-blocker filter lists. This silently broke live captioning and voice commands for any visitor running a common ad blocker, while every other feature worked normally — making it look like a feature bug rather than a network-level block. |
| **Server-side configuration** | `test_configuration.py` (28 tests), built on `Settings(_env_file=None)` so the tests see only defaults and an explicitly controlled environment, never a developer's `.env` — the exact ambient dependency that already caused one real CI failure. Covers: startup **fails loudly** when any of the four required settings is missing (parametrised, and the error names the missing setting); risky features (`streaming_enabled`, `voice_command_embedding_matching_enabled`) **default to off**, pinned as a documented trap since leaving them unset yields a server whose microphone features silently do nothing; the transcriber defaults to the fake backend so a fresh checkout never tries to load a 1 GB model; environment overrides work case-insensitively; an unrelated environment variable is ignored while a garbage port number is rejected; `CORS_ORIGINS` parsing tolerates whitespace and stray commas; relative paths resolve from the repository root **not the current directory** (verified by changing the working directory); absolute paths are respected; a missing local model path falls back to the raw value so a hub id like `small` still works. **CORS on the live app:** a configured origin is allowed; four unlisted origins are refused, including the lookalike `https://sinhaspeech.vercel.app.evil.example` (which a naive `startswith` check would let through) and `null`; the wildcard origin is asserted never to be configured alongside credentials. **Worker configuration:** the transcriber factory honours the configured backend (whitespace and case tolerant) and rejects a typo such as `wisper` with an error naming it instead of silently falling back. |
| **Oracles** | Direct observation: browser console network panel, comparing behaviour with the extension enabled vs. disabled. For the server side: exceptions raised, parsed values, and the presence or absence of the `access-control-allow-origin` response header. |
| **Required Tools** | pytest and pydantic-settings for the server side; Firefox + uBlock Origin (found the domain-blocking defect); no systematic browser matrix yet — that needs Playwright (Appendix A). |
| **Success Criteria** | Not yet defined. |
| **Special Considerations** | Coverage to date is effectively "one browser, one OS, found by accident." The fix (a real, non-DNS-abuse-associated domain) removes the specific defect found but does not constitute configuration testing — the matrix below is still untested. |

**Now covered** by `frontend/e2e/browser-matrix.spec.js` across four projects (Chromium,
Firefox, WebKit, Pixel 7 viewport): sign-in and deep-link reload in each engine, a console-error
check, a recorded media-API support matrix, and responsive layout (no horizontal scroll, minimum
touch-target size). Suspicion about `MediaRecorder` was correct — see Appendix A.2, finding 5.

**Not yet done:**
- Real devices and real Safari. Playwright's WebKit is Safari's engine, not Safari, and cannot
  certify iOS.
- Edge specifically (Chromium-based, so largely covered by proxy).
- With/without common ad blockers, now that one specific case is known to matter.
- Throttled/slow network conditions, relevant given the deployment's `us-east-1` region vs. an
  expected Sri Lanka–based user base (~250 ms round trip vs. ~50 ms from a Mumbai region).

---

## 4. Deliverables

### 4.1 Test Evaluation Summaries

**Test logs.** `pytest -v` output, produced on every local run and on every GitHub Actions run
(`.github/workflows/ci.yml`). CI runs on every push to `main` and every pull request, needs no
secrets, and starts a disposable Postgres 17 container so the database-dependent tests run
hermetically rather than against shared state.

**CI structure:**

| Job | Steps |
|---|---|
| Backend tests | Start Postgres 17 (health-checked) → `alembic upgrade head` → `pytest` (287 tests, 1 expected failure) |
| Frontend lint and build | `npm ci` → oxlint → production build |

There is deliberately no CD wired to this — deployment stays a manual, deliberate action for
both halves (`npx vercel --prod` for the frontend; `git pull && docker compose up -d --build`
on the server) because a backend restart costs ~70 seconds of downtime while the Whisper
models reload, which should never be triggered automatically by an unrelated commit.

**Deployment smoke tests.** `test/deployment/test_smoke.py`, 11 tests, run manually against the
live system after any deploy (`pytest test/deployment/ -v`). Deliberately excluded from CI —
`pytest.ini` scopes `testpaths` to `test/backend` only — since a smoke test failing because the
EC2 instance happens to be stopped should never mark a code commit as broken. Covers backend
reachability and TLS validity, database connectivity, the full auth flow, CORS configuration
against the deployed frontend's real origin, that the deployed JS bundle was built against the
correct API URL (Vite bakes this in at build time — a build without it looks fine and fails
every request), and that the WebSocket path accepts an authenticated session and rejects a bad
token.

**Model evaluation artifacts.** WER/CER per fine-tuning run and TF-IDF/KMeans error-cluster
analysis live in `ErrorAnalysis/`, one subfolder per run, each split into `error_analysis/`
(clusters, confusions, per-sample severity) and `run_summary/` (training curves, predictions,
wandb exports). `command_embedding_similarities_en.csv` compares four similarity techniques
(cosine, Euclidean, Manhattan, Pearson) across 435 real recording pairs.

### 4.2 Reporting on Test Coverage

No code-coverage percentage is currently measured — `pytest-cov` is not installed. Coverage is
reported here qualitatively, by technique (§3.1's status column) and by file (§3.1.1–3.1.2's
technique tables), rather than as a line/branch percentage. Adding `pytest-cov` to the CI
backend job is a low-effort next step that would make this section quantitative.

A generic report for each CI run contains, per the template's suggested fields: date (commit
timestamp), triggering user, number of tests executed, pass/fail count, and — via the GitHub
Actions log — the specific failing assertion when a run is red. This report itself is
regenerated whenever a new testing technique is substantially implemented rather than on a
fixed cadence.

---

## 5. Risks, Dependencies, Assumptions, and Constraints

| Risk | Mitigation Strategy | Contingency (Risk is realized) |
|---|---|---|
| **Local test runs are not hermetic.** `test_voice_enrollment.py`, `test_api_access_control.py`, `test_api_quiz_lifecycle.py`, `test_db_integrity.py`, `test_api_transcripts.py`, and `test_api_upload_pipeline.py` write to the real shared development database and read the local `.env` (not just CI's clean environment). This already caused one real incident: two wake-gate tests passed locally only because a local `.env` flag was `true`, while the flag defaults to `false` — CI, correctly, had none and failed both tests on their very first run. | CI now runs against a disposable Postgres 17 container with no `.env`, so it is the authoritative hermetic check regardless of what passes locally. | Before trusting a local-only green run, re-run against CI's exact conditions (`VOICE_COMMAND_EMBEDDING_MATCHING_ENABLED=false pytest -q`, or push and check Actions) rather than assuming local == correct. Verified: the full 287-test suite passes identically under this condition and on a clean rerun (no test-order or leftover-state pollution observed). |
| **`process_next_job()` operates on the whole shared job queue, not a test-scoped one.** It always claims the oldest `QUEUED` row across the entire `transcription_jobs` table — a test that called it without checking for pre-existing real jobs could silently claim and "complete" someone else's in-flight upload with fake canned text. | `test_api_upload_pipeline.py`'s tests that call `process_next_job()` first check the real queue is empty and **skip** (never force) if it is not — see `_require_empty_queue()`. | If this guard is ever removed or bypassed, a real user's queued transcription could be silently corrupted; treat any change to `_require_empty_queue()` as a change worth reviewing carefully, not routine cleanup. |
| **Dependency versions are unpinned.** `backend/requirements.txt` has zero version constraints, so the same commit can install different library versions in different environments — measured directly: local venv had `torch 2.13.0`/`transformers 5.15.1`; the deployed server and CI both had `torch 2.9.1`/`transformers 5.17.0` at time of writing. | Verified the full suite passes on the newer versions (harmless today). | `pip freeze > requirements.txt` or adopt a lockfile so builds become reproducible; a future library release could otherwise break the deployment with no code change and no warning. |
| **Fixture teardown order matters and is easy to get wrong.** `quizzes.created_by` and `quiz_submissions.student_id` are `ON DELETE RESTRICT` (by design — see §3.1.1), so a test fixture that creates a quiz or submission and then tries to delete the owning user in the usual order raises `IntegrityError` during teardown, not during the test itself, which is a confusing place to debug. | `conftest.py`'s `_delete_user` now explicitly deletes a fixture user's answer submissions, quiz submissions, and quizzes before deleting the user row — verified this eliminates the teardown errors that appeared before the fix. | Any new fixture that creates rows with a `RESTRICT` foreign key to the user must extend this cleanup, or reuse `_delete_user` rather than deleting the user directly. |
| **Concurrent streaming capacity is roughly one user.** Measured on a 2-CPU stand-in: one streaming user sees ~5s command latency, two see ~23s, and at three or more most commands never arrive within 30s (Appendix A.3). The REST API is unaffected; this is CPU-only Whisper inference, serialised behind one shared transcriber. | The number is now known rather than assumed, and the harness (`test/load/locustfile.py`) can re-measure it on any target. | The classroom scenario in §1 - a lecturer captioning live while several students use voice commands - is not achievable on a 2-vCPU instance as built. Plan on a GPU instance, a smaller model, or one streaming user at a time; and re-measure on the real instance before relying on these figures. |
| **A free dynamic-DNS domain silently breaks core features for a subset of users.** `sinhaspeech.duckdns.org` is blocked by uBlock Origin and likely other ad blockers, breaking live captioning and voice commands with no visible error — found during deployment, documented in §3.1.8. | Move to a paid, non-abuse-associated domain (`sinhaspeech.me`, already owned). | Until migrated, warn anyone demoing or evaluating the system to disable ad blockers first. |
| **Backup restoration is now proven, but only on the scratch stack.** | `test/failover/test_failover.py` restores a `pg_dump` into a separate database and compares every table's row count against the original. | The production database's backups have still never been restored on the production host; the procedure is proven, this particular data is not. |
| **Frontend test coverage is real but narrow.** Component tests now cover `useVoiceCommands.js`, the accessibility controls and the voice meter, and browser tests cover login, navigation and role boundaries — but the quiz answer flow, the transcript editor and live transcription have none. | §3.1.3, Appendix A. `npm test` runs in CI, so a regression in the covered units fails the build; the first axe run found a real WCAG AA contrast failure on the login button, now fixed. | Changes to the uncovered pages still rely on manual testing and lint/build success, neither of which catches a behavioural regression. |
| **Demo accounts use a well-known password** (`demo123`), and the API is now publicly reachable. | Acceptable for a project demo; access control tests (§3.1.6) confirm role boundaries hold even if credentials are known. | Rotate demo account passwords before any evaluation where credential secrecy matters. |

---

## 6. References

- pytest — https://docs.pytest.org/
- pytest-asyncio — https://pytest-asyncio.readthedocs.io/
- FastAPI `TestClient` (Starlette) — https://fastapi.tiangolo.com/tutorial/testing/
- PyJWT — https://pyjwt.readthedocs.io/
- GitHub Actions — https://docs.github.com/actions
- Alembic — https://alembic.sqlalchemy.org/
- Vitest (§3.1.3) — https://vitest.dev/
- React Testing Library (§3.1.3) — https://testing-library.com/react
- Playwright (§3.1.3) — https://playwright.dev/
- Locust (§3.1.5) — https://locust.io/
- axe-core, via jest-axe and @axe-core/playwright (§3.1.3) — https://github.com/dequelabs/axe-core
- `docs/DEPLOYMENT.md` — this project's own AWS EC2 + Vercel deployment guide, referenced
  throughout §3.1.7 and §5
- `ErrorAnalysis/` — this project's own model-evaluation artifacts, referenced in §3.1.4 and §4.1


---

## Appendix A. Implementation status and findings

The three techniques that §3.1 listed as 🔴 not started — UI/accessibility (§3.1.3), load
(§3.1.5) and failover/recovery (§3.1.7) — are now implemented. This appendix records what
exists, what it found, and what remains unverified.

### A.1 What was added

| Area | Where | Run it with |
|---|---|---|
| Component + accessibility tests | `frontend/src/**/*.test.jsx`, `src/test/setup.js` | `cd frontend && npm test` |
| Browser end-to-end + axe | `frontend/e2e/app.spec.js`, `playwright.config.js` | `npm run test:e2e` |
| Cross-browser matrix (4 engines) | `frontend/e2e/browser-matrix.spec.js` | `npm run test:e2e -- --project=webkit` |
| Load generation | `test/load/locustfile.py` | `locust -f test/load/locustfile.py --host http://localhost:8001` |
| Failover / recovery | `test/failover/test_failover.py` | `pytest test/failover -v` |
| Disposable test stack | `docker-compose.scratch.yml` | see A.4 |

**Frontend unit (43 tests, in CI).** `useVoiceCommands` — connect URL and token encoding, the
COMMAND start message, `command`/`command_maybe` forwarding, every `getUserMedia` error mapped
to its message, `stop()` → `session_end`, unexpected close vs. close after stop, unmount
cleanup, and the StrictMode abort that must *not* surface a false error.
`AccessibilityControls` — axe, keyboard-only operation, persistence, corrupt storage.
`VoiceMeter`. **The quiz answer flow** (18 tests) — the journey the product exists for: MCQ
selection by voice, out-of-range and clear commands, navigation guards (no skipping an
unanswered question, no wrapping past the first), the submit confirmation and its command
priority, and the same journey completed by keyboard alone. Two deliberate mutations of the
command handler were each caught, so these detect regressions rather than merely exercising
code.

**End-to-end (60 tests = 15 x 4 browser projects, not in CI).** Runs the built app against the
mock API, so no backend is needed: login validation, keyboard-only sign-in, protected-route
redirect, a student blocked from teacher pages, axe WCAG A/AA on login/dashboard/settings,
high-contrast persistence across a reload, plus the cross-browser checks in §3.1.8. All 60 pass
on Chromium, Firefox, WebKit and a Pixel 7 viewport.

**Failover (6 tests, not in CI).** A `pg_dump` restored into a scratch database and compared
table by table; the database stopped mid-request and the API recovering with no restart
(`pool_pre_ping`, now proven end to end); the backend killed and returning by itself
(`restart: unless-stopped`); an abruptly dropped WebSocket in both modes; and per-user session
slots not leaking across repeated drops.

### A.2 Findings

**1. The shared VAD was not thread-safe — it could crash the whole server.** Every streaming
session shares one Silero model and calls `vad.analyze` on a worker thread via
`asyncio.to_thread`. Unserialised, concurrent sessions corrupted its internal state:
`RuntimeError: select(): index 1 out of range for tensor of size [1, 64]` at best, and a
*fatal interpreter crash* — not an exception — at worst, which is what a load run actually
produced. Fixed with a lock in `VoiceActivityDetector` (VAD is a few milliseconds, so the cost
is negligible). Regression test: `test/backend/test_vad_concurrency.py`, which reproduced the
crash reliably before the fix.

**2. A vanishing client aborted the server's own cleanup.** Every notification the streaming
route sends after an utterance is best-effort, but each was wrapped in `except RuntimeError`
only — which covers a *graceful* close. A client that simply disappears (dropped wifi, closed
tab, or the route change `useVoiceCommands.js` says unmounts a listening session constantly)
raises `WebSocketDisconnect(1006)`, and under load uvicorn's `ClientDisconnected` (an
`OSError`). Either escaped, aborting `_finalize_remaining_buffer` mid-cleanup and surfacing as
an unhandled ASGI error. Fixed by naming the condition once (`_CLIENT_GONE`) and applying it to
all eight send sites. Regression tests: `test/backend/test_streaming_disconnect.py` (17 cases,
unit speed) and the failover suite's COMMAND case, which was verified to fail against the
unfixed route.

Damage was bounded and worth stating precisely: dictated text is persisted *before* the send,
and the session slot is released in a `finally`, so no note was lost and no slot leaked. The
defect was that the route's stated intent — "a dead client is never fatal" — did not hold for
the most common way a client dies.

**3. Inference threads were not matched to the CPU budget — a 2.8x latency penalty.**
`WhisperModel` was constructed without `cpu_threads`, so CTranslate2 started one thread per
*visible* core. On a dedicated host that is correct. Wherever the CPU budget is smaller than the
machine — a container with `cpus:` set, or a shared instance — those threads thrash: measured on
a 2-CPU-capped container that still saw all 12 host cores, the same 2.25s clip took **11.4s** to
transcribe at the default and **4.1s** with threads set to 2. End to end, command latency fell
from 12-15s to 3.6-5.2s. Added `streaming_cpu_threads`, defaulting to `0` — CTranslate2's
existing behaviour, so nothing changes for the current deployment, where the 2-vCPU instance
sees exactly 2 cores. It exists so a CPU-limited container can be told the truth about its
budget. This was found only because load testing forced the question of where the time goes.
**Confirmed directly on the real instance 2026-09-20** (`EC2_TEST_RUNBOOK.md` Part 1):
`nproc` reports 2, the backend container's cgroup quota is `max 100000` (unrestricted), and
`multiprocessing.cpu_count()` inside the container also reports 2 — cores-seen matches the
real budget exactly, so no override is needed on this instance today; the setting exists for
the day the instance or its container limits change.

**4. Live captioning and voice commands were refused on Safari's engine, for an API they
never use.** Both `useVoiceCommands` and `LiveTranscription` gated on `window.MediaRecorder`
before opening a session — but neither constructs one. Their capture pipeline is
`getUserMedia → AudioContext → ScriptProcessor`. Playwright's WebKit reports no MediaRecorder
(and Safari had none at all before 14.1), so on those browsers a student was told the two
features this application exists for were "not supported", on a browser that runs them
perfectly well. A false-negative capability check is the worst shape this bug could take: it
fails silently, looks deliberate, and disables an accessibility feature for the people who
most depend on it. Both guards now check `getUserMedia` and `AudioContext`, the APIs actually
used. `AudioRecorder` and voice enrolment, which genuinely do construct a `MediaRecorder`,
keep theirs. Found by the browser matrix on its first run — precisely the defect class §3.1.8
predicted but had never been able to look for.

**5. Insufficient colour contrast on the primary button.** axe measured white on `#a78bfa` at
2.72:1, against the 4.5:1 WCAG AA minimum — on the login button, the first control every user
meets. Fixed by using `--teal-dark` (5.7:1) for `.button--primary`. The e2e axe checks on
login, dashboard and settings now pass unconditionally.

**6. The English past tense of a destructive command scored above the fuzzy-match bar.**
`test_command_safety.py`'s adversarial sweep (§3.1.2's item 9) found that "deleted" — plausible
in ordinary dictation, e.g. "I deleted my notes" — scored 92.3 against `delete` under the fuzzy
skeleton matcher, above the then-current `voice_command_destructive_threshold` of 90.0. Every
other near-miss tested (21 phrases, Sinhala and English) topped out at 83.3, a comfortable gap,
so this was a real, isolated defect rather than the matcher being generally too loose. Fixed by
raising the threshold to 95.0 — still well below every genuine exact-phrase match (100) and
comfortably above every tested near-miss, including "deleted". Full backend suite (364 tests)
re-run clean after the change. Unlike the embedding thresholds elsewhere in this file, this
number is not yet backed by a large real-recording sample — it should be re-validated the same
way (`scripts/validate_command_embeddings.py`-style analysis) once more real usage data exists.

### A.3 Load testing

`test/load/locustfile.py` has two user classes: `ApiUser` (ordinary authenticated browsing) and
`StreamingUser` (one COMMAND-mode session, streaming real 16 kHz clips from
`storage/voice_samples` at true real-time pace, 250 ms per chunk, as the browser does, with a
mic-like noise floor between words rather than digital silence).

**Measured on the scratch stack** (2 CPUs, `STREAMING_CPU_THREADS=2`, wake gate off), latency
being the time from the end of the spoken clip to the server's command verdict:

| Concurrent streaming users | Verdict latency (mean) | Timeouts (30s) |
|---|---|---|
| 1 | 5.4s | 0% |
| 2 | 23.2s | 0% |
| 3 | 29.6s | 56% |
| 5 | 24.5s | 55% |

**The ceiling is between one and two concurrent streaming users on a 2-CPU budget.** One user
is workable; two already push latency past 20 seconds; three or more and most commands never
arrive within 30 seconds. The REST endpoints are unaffected — they stayed in the tens of
milliseconds throughout — so this is entirely the cost of CPU-only Whisper inference, which is
serialised behind a single shared transcriber. No crashes or restarts occurred at any level
once the VAD defect in A.2 was fixed.

Note what this means for the classroom scenario in §1: live captioning for a lecturer plus
voice commands for several students concurrently is not achievable on a 2-vCPU instance as
built. The options are a GPU instance, a smaller/faster model, or accepting one streaming user
at a time.

Two things it cannot tell you, by construction:

- It measures the *request path*, not transcription quality. It cannot say whether accuracy
  degrades under load, only whether responses arrive and how late.
- Production requires the "zimi" wake word, so bare command clips are correctly ignored and
  never answer. The scratch stack disables the gate (`VOICE_COMMAND_WAKE_REQUIRED=false`) so
  latency is measurable at all. Against any server with the gate on, `ws speech->verdict` will
  simply time out — that is the gate working, not a failure.

It has one known rough edge: at five users a few sessions ended with a `JSONDecodeError`
from the harness's own receive loop rather than a server error. That is the harness, not the
server, and it does not affect the latency figures above.

Never point it at the live deployment casually: it competes for the same two vCPUs a demo needs
and writes real rows.

### A.4 The scratch stack

`docker-compose.scratch.yml` is a disposable stack for exactly this kind of testing: its own
compose project (`dse-scratch`), its own database and volumes, and different host ports (5433,
8001), so nothing it does can touch dev data or the running dev containers. It reuses the
already-built backend image, bind-mounts the working-tree source, and is capped at 2 CPUs to
resemble the production instance — a load figure measured on all 12 host cores would mean
nothing.

```bash
docker compose -f docker-compose.scratch.yml up -d
docker compose -f docker-compose.scratch.yml exec backend alembic upgrade head
docker compose -f docker-compose.scratch.yml exec backend python -m scripts.seed_users
pytest test/failover -v
docker compose -f docker-compose.scratch.yml down -v   # throw it all away
```

The failover tests refuse to run against anything that is not this stack.

### A.5 Still unverified

- **Real hardware.** Everything above ran on a 12-core laptop with the backend capped at 2 CPUs.
  That approximates the 2-vCPU EC2 instance; it is not the same as measuring it.
- **The voice-enrolment and voice-sample HTTP routes** (36% and 33% covered) — the service
  beneath them is at 100%, but the endpoints themselves are barely exercised. See A.6.
- **The transcript editor and live transcription still have no component tests.** The quiz
  answer flow, the voice-command hook, the accessibility controls and the voice meter do.
- **Playwright's Chromium download repeatedly stalled here**, so both Chromium-backed projects
  fall back to an already-installed binary via `PW_CHROMIUM_PATH`. Firefox and WebKit
  downloaded normally. On a normal connection `npx playwright install` is enough and the
  variable is unnecessary.
- **Real devices and real Safari.** Playwright's WebKit is Safari's engine, not Safari; it
  found a genuine defect but cannot certify iOS.
- **Locust is not pinned.** It was installed ad hoc and is absent from `backend/requirements.txt`,
  which still has no version constraints at all (see §5).
- **Multi-worker deployment.** `_active_sessions` is per-process, so the session cap and these
  leak tests only hold on a single worker, as the code's own comment notes.

### A.6 Coverage audit

A full pass over the source tree, measured with `pytest --cov` rather than inferred:

**Backend: 84.16% of statements (2474 total, 392 uncovered) as of 2026-09-20, up from 79% —
measured via `pytest --cov=app --cov-report=term`, and now gated in CI at a 75% floor
(`.github/workflows/ci.yml`, `--cov-fail-under=75`).** Fully covered: every model and
schema, the streaming buffer, command matching, command resolution, the voice-enrolment
*service*, and — closed in this pass — `services/streaming_persistence.py` (33% → 100%,
`test_streaming_persistence.py`) and the WebSocket-auth branches of `api/dependencies.py`
(58% → 87%, `test_ws_auth_dependency.py`; `get_current_user_ws` needed no real socket, just
a stub object exposing `.query_params.get()`). The uncovered remainder falls into two groups.

*Needs a loaded model or a real server — the manual work in `EC2_TEST_RUNBOOK.md`:*

| Module | Cover | Why |
|---|---|---|
| `transcribers/whisper.py` | 15% | Loads a real 1.2 GB Whisper model |
| `transcribers/speak_asr.py` | 29% | Calls an external ASR service |
| `streaming/inference.py` | 49% | Model construction and warm-up |
| `streaming/embeddings.py` | 68% | Needs the embedding model |
| `api/routes/streaming.py` | 71% | The rest needs a real WebSocket, covered by `test/failover/` |

*Genuine gaps, testable without a model — closed 2026-09-20 (`TEST_GAPS_README.md` Task 3),
except `streaming_persistence.py`, which still needs a live socket:*

| Module | Cover (was → now) | Status |
|---|---|---|
| `api/routes/voice_samples.py` | 33% → **91%** | `test_api_voice_enrollment.py`: every endpoint's happy path, 401, 404, 400. |
| `api/routes/voice_enrollment.py` | 36% → **91%** | Same file — all 5 endpoints driven through `TestClient`, embedding call monkeypatched so no model loads. |
| `api/routes/media.py` | (untested route) → **88%** | `test_api_media.py`: owner, non-owner, teacher-via-LECTURE, teacher-blocked-on-NOTE. |
| `services/media_access_service.py` | 56% → **94%** | Same tests, service-layer ownership/permission branches. |
| `services/streaming_persistence.py` | 33% → **100%** | Turned out to need no socket at all - plain functions called directly in `test_streaming_persistence.py`. `test/failover/` still covers the real-socket path separately. |
| `api/dependencies.py` | 58% → **87%** | `get_current_user_ws` fully covered via `test_ws_auth_dependency.py`. The remaining 13% is `require_roles` (lines 86-96) — **dead code**, confirmed via `grep`: nothing calls it, every route uses its own `require_teacher`/`require_student` in `routes/quiz.py` instead. Worth a cleanup PR to remove it rather than a test. Line 33 (`get_current_user`'s own non-bearer-scheme check) is separately unreachable: `HTTPBearer(auto_error=False)` already returns `None` for any non-Bearer scheme before that line can run, confirmed directly against `HTTPBearer.__call__`. |

**Frontend.** Four units have component tests — `useVoiceCommands`, `AccessibilityControls`,
`VoiceMeter` and the quiz answer flow. Untested: `LiveTranscription`, `TranscriptEditor`,
`AudioRecorder`, `FileUpload`, `useTranscriptionJob`, `AppShell`, `AuthContext`,
`ToastContext`, and every page other than the quiz flow. Login, navigation, role boundaries
and accessibility on three pages are covered end-to-end instead, across four browsers.

**A note on suite reliability.** The e2e suite is pinned to one worker. Four browser engines in
parallel exceeded available memory on the development machine and produced a timeout in a
different test on most runs — never an assertion failure. Two workers failed 3 of 5 runs; one
worker passed 2 of 2 at the same wall-clock time. This is recorded because an intermittently
red suite teaches people to ignore it.

### A.7 CI

`npm test` runs in the existing frontend job alongside lint and build. Playwright, Locust and
the failover suite stay out of CI deliberately: they are slow, they need Docker or a browser,
and a load test that fails because a shared runner was busy says nothing about the commit. They
are run by hand, the same way the deployment smoke tests in §4.1 are.
