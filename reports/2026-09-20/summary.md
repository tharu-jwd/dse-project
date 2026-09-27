# Test run — 2026-09-20

| Step | Tests | Verdict | Seconds |
|---|---|---|---|
| Backend (pytest) | 333/333 | PASS | 42 |
| Frontend (vitest) | 43 passed | PASS | 5 |
| Playwright (e2e) | 60/60 | PASS | 143 |
| Deployment smoke | 11/11 | PASS | 14 |

*(The 333/333 above predates Task 4's 3 new stuck-job-recovery tests, run separately;
the full backend suite is 336 with them included — see the task table below.)*

## Test-gap sprint run log

Per `TEST_GAPS_README.md`. Tasks 1, 2, 3, 4, 5, 6, 8, 9, 10 done; only 7 (mobile audit)
skipped.

| Task | What was added | Tests added | Result | Time |
|---|---|---|---|---|
| 1. Isolated test stack | `test/conftest.py` (`SINHASPEECH_TEST_DB_URL` → `POSTGRES_*` translation, loaded before `app.db.session`), `scripts/test-isolated.sh` (scratch DB up → migrate → pytest → teardown) | 0 (infra) | PASS — 333 passed, 0 skipped against an empty scratch DB, no `.env` | ~20 min |
| 2. One-command run + evidence | `scripts/test-plan.sh` — backend (via Docker image), frontend (vitest), Playwright, deployment smoke; each independently skippable; writes `reports/<date>/` with logs + JUnit XML + summary table | 0 (infra) | PASS — produced this folder | ~15 min |
| 3. Voice-enrolment & voice-sample API tests | `test/backend/test_api_voice_enrollment.py`, `test/backend/test_api_media.py` (media access was folded in alongside, same gap category) | 27 | PASS — 27/27. Coverage-percentage verification hit an unrelated local pip/numpy double-import quirk installing `pytest-cov` at runtime; not chased further, but every endpoint's happy path, auth-required, and 404 cases are exercised | ~30 min |
| 4. Stuck-job recovery | `claim_next_job()` in `transcription_processor.py` now also reclaims a `PROCESSING` job whose `started_at` is older than a new `transcription_stuck_job_timeout_minutes` setting (default 30); `test/backend/test_stuck_job_recovery.py` | 3 | PASS — reclaim-when-stale, leave-alone-when-fresh, and missing-media-fails-without-blocking-the-queue all verified | ~20 min |
| 5. Destructive-command safety | `test/backend/test_command_safety.py` — 21 near-miss phrases (SI+EN) against `match_command`, wake-word gating checks, and a printed score-margin table | 28 | PASS after a real fix (see D-1 below) | ~20 min |
| 6. Immersion / leaked internals | `frontend/e2e/immersion.spec.js` — visits every student-facing page (login, dashboard, transcripts, quizzes, settings, help) plus a real transcript detail and quiz answer page, checks rendered text for leaked tokens (`undefined`/`null`/`[object Object]`/`Traceback`/`isCorrect`/`transcriptId`), raw UUIDs, and unformatted ISO timestamps | 8 | PASS — no leak found | ~15 min |
| 7. Mobile layout audit | — | 0 | SKIPPED (lowest priority per README's own ordering) | — |
| 8. Accessibility on quiz/editor pages | `frontend/e2e/quiz-and-editor-a11y.spec.js` — axe (WCAG A/AA) on the quiz answer page, transcript editor page, and live transcription (new note) page, plus a keyboard-only MCQ-selection walkthrough | 4 | PASS — 3/3 axe checks clean, 0 violations; keyboard walkthrough skips (no MCQ question exists in the demo student's current quiz list — a data-state skip, not a test bug) | ~20 min |
| 9. Clean-install & compose validation | `test/backend/test_clean_install.py` — migrations + `seed_users` + demo login against a genuinely fresh database (not the shared dev DB, which has never been empty); `docker compose config` on all 3 compose files; every `${VAR}` they reference checked against `.env.example` | 5 | PASS — no start-up bug found this run; skips cleanly without Docker | ~15 min |
| 10. Coverage gate + secret scan in CI | `.github/workflows/ci.yml` — backend job now runs `pytest --cov=app --cov-fail-under=75`; new `secret-scan` job runs `gitleaks/gitleaks-action@v2` over full commit history (`fetch-depth: 0`) on every push/PR | 0 (infra) | PASS — measured 82.78% locally (up from 79%), comfortably above the 75% gate. gitleaks not run locally (not installed); confirmed no `.env` was ever committed via `git ls-files` / history search | ~15 min |
| 11. Follow-up coverage push (user-requested, beyond the README) | `test/backend/test_streaming_persistence.py` (streaming_persistence.py: 33%→100%), `test/backend/test_ws_auth_dependency.py` (dependencies.py: 58%→87%, `get_current_user_ws` tested directly with a stub object, no socket needed) | 14 | PASS. Found 2 more small things: `require_roles()` in dependencies.py is dead code (nothing calls it — routes use their own `require_teacher`/`require_student`); `get_current_user`'s own non-bearer-scheme check (line 33) is unreachable — `HTTPBearer(auto_error=False)` already filters non-Bearer schemes before that line runs. Neither changed — documented, not fixed. | ~20 min |

**Totals:** 80 new backend tests, full backend suite now 378 passed, 5 skipped (was 305 at the
start of this sprint); 12 new Playwright tests (8 immersion + 4 quiz/editor a11y), full e2e
suite 71 passed + 1 skipped (data-state, not a bug) + 1 confirmed-flaky unrelated pre-existing
test (passes in isolation); backend statement coverage 79% → **84.16%**, now CI-gated at 75%;
CI also gained a full-history secret scan (gitleaks). Two dead-code findings documented but not
fixed (`require_roles`, an unreachable scheme check in `get_current_user`).

### Defects found

| # | Found by | Defect | Cause | Fix / status |
|---|---|---|---|---|
| D-1 | Task 5, `test_near_miss_scores_stay_meaningfully_below_the_destructive_bar` | The English past tense "deleted" (plausible in ordinary dictation — "I deleted my notes") scored 92.3 against the fuzzy destructive-command bar of 90.0, above threshold, and would have executed a real `delete` command | `voice_command_destructive_threshold` (90.0) was closer to a false positive than any other near-miss tested (next closest: 83.3) | **Fixed** — raised to 95.0 in `backend/app/core/config.py`, comfortably above every tested near-miss and still below every genuine exact-phrase match (100). Full backend suite re-run clean after the change (364 passed) |
| D-2 | Task 4 (found while writing the test, not by a running assertion) | `create_transcription_job()`'s duplicate-title cancellation (`_cancel_pending_job_with_title`) silently deletes an earlier still-pending job for the same user+title — not itself a bug (intended dedup behaviour), but easy to trip over in a test or a real double-submit-with-same-title from the UI | Existing, intentional behaviour; flagged here because it was non-obvious and cost real debugging time | **Open** — not changed; documented in `test_stuck_job_recovery.py`'s `_upload()` docstring so the next test author doesn't hit the same trap. Worth a UX check: does the frontend ever let a student re-upload under an identical in-flight title unintentionally? |

