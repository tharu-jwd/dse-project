# SinhaSpeech — Master Test Plan

Last updated 2026-09-19. Structured against the Master Test Plan template (Rational Unified
Process format). Every "Status" line below reflects what actually exists and passes as of
this date — not what is planned — with plans separated out explicitly in §3.1's tables.

**At a glance:**

| Technique (§3.1) | Status |
|---|---|
| 3.1.1 Data & Database Integrity | 🟡 Partial |
| 3.1.2 Function Testing | 🟡 Partial |
| 3.1.3 User Interface Testing | 🔴 Not started |
| 3.1.4 Performance Profiling | 🟡 Partial |
| 3.1.5 Load Testing | 🔴 Not started |
| 3.1.6 Security & Access Control | 🟢 Implemented |
| 3.1.7 Failover & Recovery | 🔴 Not started |
| 3.1.8 Configuration Testing | 🟡 Partial |

```
189 backend tests   (pytest, test/backend/)
 11 deployment smoke tests   (pytest, test/deployment/, run manually against the live system)
  0 frontend tests   (lint + build only)
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

**Status: 🟡 Partial.**

| Technique Objective | Exercise the database and its access methods independent of the UI — schema creation, foreign keys, and per-row correctness — to observe data corruption or incorrect persistence. |
|---|---|
| **Technique** | `test_voice_enrollment.py` (18 tests) inserts and deletes real rows against a live Postgres instance: samples accepted/rejected by similarity, per-language isolation, cascade delete on user removal, unknown-id rejection. `test_db_integrity.py` (5 tests) exercises constraint and foreign-key behaviour directly against Postgres, independent of the service layer: `uq_command_enrollments_slot` rejects a duplicate `(user, command, language, sample_index)` row while correctly allowing the same `sample_index` under a different language; deleting a user cascades their voice enrollments (`ON DELETE CASCADE`); deleting a teacher who owns quizzes is *rejected* by the database (`ON DELETE RESTRICT` on `quizzes.created_by`); deleting a quiz cascades its questions (`ON DELETE CASCADE` on `questions.quiz_id`). In CI, `alembic upgrade head` runs against a disposable Postgres 17 service container on every push, so schema creation itself is verified on every commit, not just once. |
| **Oracles** | Direct row inspection via SQLAlchemy queries in the test body; `pytest.raises(IntegrityError)` for constraint violations. |
| **Required Tools** | pytest, SQLAlchemy, Postgres 17 (Docker for CI, dev instance for local runs) |
| **Success Criteria** | All 23 tests pass; migrations apply cleanly to an empty database. |
| **Note on running these locally** | Because they write to the shared dev Postgres instance, cleanup is intentional and verified rather than assumed: fixture teardown (`conftest.py`'s `_delete_user`) explicitly removes quiz submissions, quizzes, transcripts, transcript segments, transcription jobs, and media files owned by a fixture user, in an order that respects each table's real `ondelete` policy, before deleting the user row itself. Confirmed by running the full suite three times in a row (including once under CI's exact environment) and checking the database directly each time for leftover fixture rows or stuck job-queue entries — none found. |
| **Special Considerations** | Locally, these tests still run against the shared dev database rather than an isolated one (see §5, risk 1) — CI is the hermetic check. Writing the RESTRICT/CASCADE tests required checking each foreign key's actual `ondelete` setting in the model file rather than assuming — the two policies are deliberately different (`RESTRICT` protects a teacher's quiz history from silently vanishing; `CASCADE` on child rows like questions/options is correct because they are meaningless without their parent). |

**Not yet done:**
- Transaction rollback on a mid-operation failure
- Concurrent writes to the same transcript
- The job queue's `SELECT ... FOR UPDATE SKIP LOCKED` claim, tested with only one worker so
  far — never verified with two workers racing for the same job

#### 3.1.2 Function Testing

**Status: 🟢 Strong across matching logic, the quiz lifecycle, transcripts, and the upload pipeline. Two narrow gaps remain (noted below).**

| Technique Objective | Exercise target functionality — navigation, data entry, processing, retrieval — via black-box interaction, verifying business rules are correctly applied for both valid and invalid input. |
|---|---|
| **Technique** | Five layers. (1) **Matching/streaming logic** — 129 tests (`test_commands.py`, `test_command_resolution.py`, `test_embeddings.py`, `test_streaming_buffer.py`, `test_streaming_commands_route.py`, `test_inference_kwargs.py`) drive the fuzzy-text matcher, the voice-embedding matcher, their combination rule, the rolling audio buffer, and the full wake-word gate end-to-end with a fake WebSocket — no real model or database. (2) **Quiz lifecycle** — 11 tests (`test_api_quiz_lifecycle.py`) drive the real FastAPI app via `TestClient`: a teacher creates a quiz, it is correctly hidden from students until published, a student answer is validated against the actual question/option it belongs to, a student's correct-answer view never leaks `isCorrect`, resubmitting updates rather than duplicates a submission, a teacher marks it, and a second teacher is confirmed unable to review a submission for a quiz they don't own. (3) **Transcripts** — 17 tests (`test_api_transcripts.py`) cover read/update/export/delete/finalize: an owner can edit and export; a second student cannot read, edit, export, or delete another student's transcript; a teacher can read *any* LECTURE transcript by design but not a student's private NOTE, and critically cannot edit a lecture they can only read (the read carve-out does not imply write access); a finalized transcript rejects further edits; renaming to an already-used title is rejected. (4) **Upload → worker pipeline** — 7 tests (`test_api_upload_pipeline.py`) drive `POST /transcriptions` through `TestClient` exactly as a browser would, then hand the queued job to the same `process_next_job()` function the real worker process loops on (using `FakeTranscriber`, the project's own fake backend for exactly this purpose) — covering upload validation (unsupported type, empty file, a note requiring audio not video), job-status privacy, and the full queue → claim → transcript-appears path end to end. (5) **API surface / auth** — 29 tests (`test_api_access_control.py`), see §3.1.6. |
| **Oracles** | Direct assertion against expected return values (`match_command()`'s score, `resolve_command()`'s outcome) and HTTP status codes / response bodies for the API layer. |
| **Required Tools** | pytest, pytest-asyncio, FastAPI `TestClient`, PyJWT |
| **Success Criteria** | All identified use-case flows for the covered areas pass with both valid and invalid input. |
| **Special Considerations** | The matching-logic tests are unusually thorough for a student project because a real bug was found and fixed here mid-project (see §5's finding write-up in the git history) — every edge case in that table is a real failure mode that was hit, not a hypothetical. Writing the quiz lifecycle tests surfaced an undocumented validation rule (`QuizCreate` requires MCQ questions to have exactly 4 options) that was not obvious from the route code alone — found by running the test and reading the resulting 422, not by reading the schema first. The upload pipeline tests run against the shared dev database and `process_next_job()` always claims the *oldest* queued job across the whole table — a naive test could accidentally claim and "complete" a real, unrelated in-flight job with `FakeTranscriber`'s canned text. Guarded against by checking the real queue is empty before those tests run and skipping (not forcing) otherwise — see `_require_empty_queue()`. |

**Not yet done — the two remaining gaps:**
- The SPOKEN-answer submission path (as opposed to MCQ) is untested — it requires a real
  transcript row owned by the submitting student; now that `test_api_transcripts.py`'s
  `_make_transcript()` helper exists, this is a small follow-up rather than a blocked one
- No test confirms a malformed payload returns 422 across the *whole* API — a few examples now
  exist (`test_malformed_payload_returns_422_not_500`, the MCQ-option-count case) but coverage
  is not systematic

#### 3.1.3 User Interface Testing

**Status: 🔴 Not started.**

| Technique Objective | Exercise navigation, form submission, and object states to observe standards conformance and target behavior; for this application specifically, exercise keyboard-only and screen-reader interaction paths since accessibility is the core requirement. |
|---|---|
| **Technique** | Not yet implemented. Planned: Vitest + React Testing Library for component-level tests, starting with `useVoiceCommands.js` (the hook where the real production bug in §5 actually lived on the client side) and the quiz/MCQ answer flow. Playwright for end-to-end flows: login → record a note → speak a command → submit. |
| **Oracles** | Rendered DOM assertions; for accessibility, `axe-core` or equivalent for WCAG contrast/label checks; manual verification of the high-contrast and text-size settings. |
| **Required Tools** | Vitest, React Testing Library, Playwright, axe-core (none installed yet) |
| **Success Criteria** | Not yet defined — pending tool installation. |
| **Special Considerations** | `frontend/package.json` currently has no test script at all — `dev`, `build`, `lint` (oxlint), `preview` only. This is the most conspicuous gap for an application whose entire purpose is usability. |

#### 3.1.4 Performance Profiling

**Status: 🟡 Partial — real numbers exist, but ad hoc rather than a repeatable suite.**

| Technique Objective | Measure response times, transaction rates, and resource usage under normal anticipated workload to verify performance requirements. |
|---|---|
| **Technique** | `backend/scripts/benchmark_command_latency.py` measures voice-command round-trip latency. Model quality is measured directly: WER/CER across 6 fine-tuning runs (see §3a in the training-side error analysis, `ErrorAnalysis/`). Container memory was measured live on the deployed server (backend ~1.2 GB, worker ~330 MB idle) during deployment. Frontend payload was measured and reduced from ~46 MB to ~1 MB of images. |
| **Oracles** | Direct measurement compared against informal thresholds (e.g. WER/CER trend across runs); no automated pass/fail gate yet. |
| **Required Tools** | The benchmark script, `docker stats`, browser DevTools network panel |
| **Success Criteria** | Not formally defined — no documented response-time budget per endpoint. |
| **Special Considerations** | All measurements to date are point-in-time and manual, not a suite that runs and fails a build. |

**Not yet done:**
- API response times (p50/p95) per endpoint
- Time-to-first-caption during live streaming
- Voice-command end-to-end latency (speak → action), as a proper automated benchmark rather
  than a one-off script run
- Memory growth over a long streaming session (buffer leak check)

#### 3.1.5 Load Testing

**Status: 🔴 Not started.**

| Technique Objective | Subject the system to varying workloads — normal, worst-case, and concurrent-user — to determine whether it continues to function correctly beyond expected maximum load. |
|---|---|
| **Technique** | Not yet implemented. Planned: Locust or k6 driving the REST endpoints; a scripted set of concurrent WebSocket streaming sessions to find the point at which transcription latency becomes unusable on the deployed 2-vCPU instance; concurrent file uploads to observe worker queue depth, since the worker processes one job at a time. |
| **Oracles** | Response-time and error-rate thresholds under load; the point of failure becomes the finding itself if no formal threshold is set yet. |
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

**Status: 🔴 Not started.**

| Technique Objective | Simulate failure conditions — power/communication interruption, database failure, incomplete transactions — and verify the system recovers to a known, correct state without data loss. |
|---|---|
| **Technique** | Not yet implemented. Planned: restore a `pg_dump` backup into a scratch database and verify row-for-row integrity (a backup that has never been restored is a hypothesis, not a backup); kill the database container mid-request and confirm the API recovers via SQLAlchemy's `pool_pre_ping` (present in code, unverified in practice); drop a WebSocket connection mid-session and confirm the partial transcript is not lost; restart the EC2 instance and confirm all four containers return automatically (`restart: unless-stopped` is set, unverified end-to-end). |
| **Oracles** | Row-count and content comparison pre/post restore; HTTP success after a simulated database restart; container status after an instance reboot. |
| **Required Tools** | `pg_dump`/`psql`, Docker, SSH access to the deployment host |
| **Success Criteria** | Not yet defined. |
| **Special Considerations** | This category is inherently somewhat destructive (killing a live container, restarting the instance) — should be run against a disposable environment or a deliberately scheduled maintenance window, not the live deployment without warning. |

#### 3.1.8 Configuration Testing

**Status: 🟡 Partial — one real defect already found this way.**

| Technique Objective | Verify correct operation across the different hardware, software, browser, and network configurations the deployed system will actually be used under. |
|---|---|
| **Technique** | Manual testing in Firefox on Linux during deployment surfaced a real, user-facing defect: `sinhaspeech.duckdns.org` is blocked outright by uBlock Origin (and, by the same mechanism, likely AdGuard, Brave Shields, and Pi-hole), because free dynamic-DNS domains are commonly abused by malware and appear on ad-blocker filter lists. This silently broke live captioning and voice commands for any visitor running a common ad blocker, while every other feature worked normally — making it look like a feature bug rather than a network-level block. |
| **Oracles** | Direct observation: browser console network panel, comparing behaviour with the extension enabled vs. disabled. |
| **Required Tools** | Firefox + uBlock Origin (found it); no systematic browser matrix yet. |
| **Success Criteria** | Not yet defined. |
| **Special Considerations** | Coverage to date is effectively "one browser, one OS, found by accident." The fix (a real, non-DNS-abuse-associated domain) removes the specific defect found but does not constitute configuration testing — the matrix below is still untested. |

**Not yet done:**
- Chrome, Edge, Safari — Safari's `MediaRecorder` support in particular differs from Chromium's
- Mobile browsers (students may use phones)
- With/without common ad blockers, now that one specific case is known to matter
- Throttled/slow network conditions, relevant given the deployment's `us-east-1` region vs. an
  expected Sri Lanka–based user base (~250 ms round trip vs. ~50 ms from a Mumbai region)
- Responsive layout across screen sizes

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
| Backend tests | Start Postgres 17 (health-checked) → `alembic upgrade head` → `pytest` (189 tests) |
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
| **Local test runs are not hermetic.** `test_voice_enrollment.py`, `test_api_access_control.py`, `test_api_quiz_lifecycle.py`, `test_db_integrity.py`, `test_api_transcripts.py`, and `test_api_upload_pipeline.py` write to the real shared development database and read the local `.env` (not just CI's clean environment). This already caused one real incident: two wake-gate tests passed locally only because a local `.env` flag was `true`, while the flag defaults to `false` — CI, correctly, had none and failed both tests on their very first run. | CI now runs against a disposable Postgres 17 container with no `.env`, so it is the authoritative hermetic check regardless of what passes locally. | Before trusting a local-only green run, re-run against CI's exact conditions (`VOICE_COMMAND_EMBEDDING_MATCHING_ENABLED=false pytest -q`, or push and check Actions) rather than assuming local == correct. Verified: the full 189-test suite passes identically under this condition and on a clean rerun (no test-order or leftover-state pollution observed). |
| **`process_next_job()` operates on the whole shared job queue, not a test-scoped one.** It always claims the oldest `QUEUED` row across the entire `transcription_jobs` table — a test that called it without checking for pre-existing real jobs could silently claim and "complete" someone else's in-flight upload with fake canned text. | `test_api_upload_pipeline.py`'s tests that call `process_next_job()` first check the real queue is empty and **skip** (never force) if it is not — see `_require_empty_queue()`. | If this guard is ever removed or bypassed, a real user's queued transcription could be silently corrupted; treat any change to `_require_empty_queue()` as a change worth reviewing carefully, not routine cleanup. |
| **Dependency versions are unpinned.** `backend/requirements.txt` has zero version constraints, so the same commit can install different library versions in different environments — measured directly: local venv had `torch 2.13.0`/`transformers 5.15.1`; the deployed server and CI both had `torch 2.9.1`/`transformers 5.17.0` at time of writing. | Verified the full 189-test suite passes on the newer versions (harmless today). | `pip freeze > requirements.txt` or adopt a lockfile so builds become reproducible; a future library release could otherwise break the deployment with no code change and no warning. |
| **Fixture teardown order matters and is easy to get wrong.** `quizzes.created_by` and `quiz_submissions.student_id` are `ON DELETE RESTRICT` (by design — see §3.1.1), so a test fixture that creates a quiz or submission and then tries to delete the owning user in the usual order raises `IntegrityError` during teardown, not during the test itself, which is a confusing place to debug. | `conftest.py`'s `_delete_user` now explicitly deletes a fixture user's answer submissions, quiz submissions, and quizzes before deleting the user row — verified this eliminates the teardown errors that appeared before the fix. | Any new fixture that creates rows with a `RESTRICT` foreign key to the user must extend this cleanup, or reuse `_delete_user` rather than deleting the user directly. |
| **Load behaviour under concurrent users is unknown.** The deployed instance has 2 vCPUs running CPU-only Whisper inference; nothing has measured what happens with 5+ simultaneous streaming sessions. | None yet — this is the primary justification for prioritising §3.1.5 next. | If a live demo or classroom session exceeds the (currently unknown) concurrency limit, transcription latency will degrade with no advance warning; the fallback is to reduce simultaneous users manually until load testing establishes a real number. |
| **A free dynamic-DNS domain silently breaks core features for a subset of users.** `sinhaspeech.duckdns.org` is blocked by uBlock Origin and likely other ad blockers, breaking live captioning and voice commands with no visible error — found during deployment, documented in §3.1.8. | Move to a paid, non-abuse-associated domain (`sinhaspeech.me`, already owned). | Until migrated, warn anyone demoing or evaluating the system to disable ad blockers first. |
| **The database has automated backups tested for creation but not restoration.** A `pg_dump` command exists in `DEPLOYMENT.md`; no restore has ever been performed. | Planned under §3.1.7. | If the production database is lost before this is verified, recovery is unproven and may fail silently or partially. |
| **No frontend tests exist for an accessibility-first application.** A regression in `useVoiceCommands.js` or the accessibility settings would currently ship undetected — this is exactly the class of bug that shipped once already (see the wake-word incident referenced above, which had a client-side half). | Planned under §3.1.3; Vitest + React Testing Library is the natural fit given the Vite toolchain already in use. | Until implemented, frontend changes rely entirely on manual testing and oxlint/build success, neither of which catches behavioural regressions. |
| **Demo accounts use a well-known password** (`demo123`), and the API is now publicly reachable. | Acceptable for a project demo; access control tests (§3.1.6) confirm role boundaries hold even if credentials are known. | Rotate demo account passwords before any evaluation where credential secrecy matters. |

---

## 6. References

- pytest — https://docs.pytest.org/
- pytest-asyncio — https://pytest-asyncio.readthedocs.io/
- FastAPI `TestClient` (Starlette) — https://fastapi.tiangolo.com/tutorial/testing/
- PyJWT — https://pyjwt.readthedocs.io/
- GitHub Actions — https://docs.github.com/actions
- Alembic — https://alembic.sqlalchemy.org/
- Vitest (planned, §3.1.3) — https://vitest.dev/
- React Testing Library (planned, §3.1.3) — https://testing-library.com/react
- Playwright (planned, §3.1.3, §3.1.8) — https://playwright.dev/
- Locust (planned, §3.1.5) — https://locust.io/
- axe-core (planned, §3.1.3) — https://github.com/dequelabs/axe-core
- `docs/DEPLOYMENT.md` — this project's own AWS EC2 + Vercel deployment guide, referenced
  throughout §3.1.7 and §5
- `ErrorAnalysis/` — this project's own model-evaluation artifacts, referenced in §3.1.4 and §4.1
