# Testing Report

Last updated 2026-09-18. Covers the four layers this project has testing for: **backend**,
**frontend**, **model (ASR) evaluation**, and — since the AWS deployment — **live deployment
smoke tests**. Frontend still has no automated test suite; that's called out explicitly
rather than skipped over.

**At a glance:**

| Layer | Tests | Runs |
|---|---|---|
| Backend unit/integration | **120** | Locally, and on every push via GitHub Actions |
| Deployment smoke | **11** | Manually, against the live system |
| Frontend | **0** | Lint + build only |
| Model (ASR) evaluation | n/a — measured, not asserted | Manually, per training run |

---

## 1. Backend — automated tests

**120 tests, all passing**, run with `pytest` from the repo root (`test/backend/`, config in
root `pytest.ini`). All but one file (`test_voice_enrollment.py`) are pure unit tests with no
network, database, or real ML model involved — fast (~7-8 seconds total) and deterministic.

```
test/backend/test_command_resolution.py   10 tests
test/backend/test_commands.py             31 tests
test/backend/test_embeddings.py           15 tests
test/backend/test_inference_kwargs.py      4 tests
test/backend/test_streaming_buffer.py     12 tests
test/backend/test_streaming_commands_route.py  30 tests
test/backend/test_voice_enrollment.py     18 tests
                                          ─────
                                          120 tests, 120 passed
```

### What each file actually protects against

| File | What it tests | Why it matters |
|---|---|---|
| `test_commands.py` | Fuzzy text matching (`skeleton()`, `match_command()`) against the Sinhala/English command vocabulary; the wake word is correctly excluded from matchable phrases; `eka`/`deka` stay cleanly separated (< 80% similarity) now that phrases aren't prefixed. | Direct regression guard for the exact bug reported during manual testing (§5) — commands scoring too low when transcribed correctly, or too high against the wrong command. |
| `test_embeddings.py` | The vector math underneath voice-fingerprint matching: pooling, L2 normalization, cosine/Manhattan similarity, `best_match()` — including zero-vector and empty-bank edge cases. | These are the functions a silent numeric bug (e.g. a NaN from dividing by zero on a silent clip) would hide in; each edge case here was a real, plausible failure mode, not a hypothetical. |
| `test_command_resolution.py` | The decision table that combines the fuzzy-text and embedding channels: execute / confirm / ignore, and the higher bar required for destructive commands (`delete`, `submit`). | This is the safety-critical logic — it's what stops an ambiguous or accidental utterance from deleting a student's work. |
| `test_inference_kwargs.py` | Whisper is only biased toward command words in COMMAND mode, never in NOTE/dictation mode. | Prevents ordinary dictation from being silently nudged toward command vocabulary. |
| `test_streaming_buffer.py` | The rolling audio buffer: correct timestamps across VAD-triggered finalizes and forced cuts, overlap handling at the boundaries. | A timestamp bug here would silently corrupt every transcript segment's start/end time — hard to spot by eye, easy to catch here. |
| `test_streaming_commands_route.py` | The WebSocket route end-to-end (with a fake socket, no real network): debouncing repeated commands, the "listening" UI signal, and — as of this session — the full wake-word gate (arm/expire/one-shot-unlock, both wake-word spellings, voice-only detection, "zimi" never saved as note text). | This is where the actual bug from manual testing was fixed and is now regression-tested: a command without a preceding wake word is provably ignored, not just "seems to work." |
| `test_voice_enrollment.py` | Enrollment: samples accepted/rejected by similarity, per-language isolation, the wake word can be enrolled and loads into the matching bank correctly. | Runs against a real (dev) Postgres database — the one file in the suite that isn't fully isolated (see §1b). |

### The database-isolation limitation — now solved in CI

`test_voice_enrollment.py` talks to the real development database when run locally
(documented in `test/backend/conftest.py`). Each test cleans up its own throwaway user, but
locally the suite still isn't hermetic and can't run without Postgres up.

**In CI this is fixed.** The workflow starts a disposable Postgres 17 service container,
applies migrations to it, and destroys it when the job ends — so every CI run begins from a
genuinely empty database. Running locally still uses the shared dev database; CI is now the
authoritative hermetic check.

---

## 1b. Continuous Integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request. It needs no
secrets and touches no infrastructure, so it is safe on pull requests from anyone.

| Job | Steps |
|---|---|
| **Backend tests** | Start Postgres 17 (health-checked) → `alembic upgrade head` → `pytest` (120 tests) |
| **Frontend lint and build** | `npm ci` → oxlint → production build |

There is deliberately **no CD**. Deployment is manual for both halves — `npx vercel --prod`
for the frontend, `git pull && docker compose up -d --build` on the server. Automating the
backend would require either opening SSH to the internet or storing a server key in the
repository, and each backend restart costs ~70 seconds of downtime while the Whisper models
reload — not something worth triggering on a README typo.

Note that `pytest.ini` sets `testpaths = test/backend`, so CI runs **only** the unit suite.
The deployment tests in §2 are excluded by design: a smoke test failing because the EC2
instance is stopped should never mark a code commit as broken.

---

## 2. Deployment smoke tests

**11 tests, all passing**, in `test/deployment/test_smoke.py`. These make real network calls
to the live frontend and backend, and they fail if the deployment is down however correct the
code is. That's the point — they answer *"is the thing I just deployed actually working?"*,
which the unit suite structurally cannot.

```bash
pytest test/deployment/ -v          # run after any deploy
```

They're excluded from the default `pytest` run and from CI. The target URLs are overridable
via `SMOKE_API_URL` / `SMOKE_APP_URL`, so the same suite can verify a future domain or a
staging environment.

| Area | What's asserted | Why it's there |
|---|---|---|
| Backend reachable | HTTPS `/health` returns healthy | Also validates the TLS certificate — `urllib` verifies by default, so a lapsed Let's Encrypt renewal fails here |
| Database | `/health/database` reports connected | The API process can be up while Postgres is unreachable; that looks fine until the first real request |
| Auth | Login returns a token; an authenticated request succeeds | End-to-end proof of the JWT path |
| Auth (negative) | An **unauthenticated** request is rejected (401/403) | A deployment that serves data to anyone is worse than one that's down |
| CORS | Backend explicitly allows the deployed frontend's origin | The single most common "deployed but nothing works" cause. CORS is browser-enforced, so `curl` passing proves nothing — it must be asserted deliberately |
| Frontend | Site serves, **and its JS bundle references the correct API URL** | Vite bakes `VITE_API_BASE_URL` in at build time; a frontend built without it looks perfectly healthy while failing every request. This downloads the real bundle and greps it |
| WebSocket | An authenticated session connects; a bad token is refused | Live captioning and voice commands run over this socket — a separate code path from the HTTP API that can fail independently (e.g. a proxy not forwarding upgrades) |

Every one of these corresponds to something that actually broke, or could silently break,
during the real deployment — see `DEPLOYMENT.md`.

---

## 3. Frontend — no automated tests

`frontend/package.json` has `dev`, `build`, `lint` (oxlint), and `preview` — **no test
script, no Jest/Vitest, no Testing Library installed.** This was a deliberate scope decision
earlier in this work, not an oversight: you asked for backend tests specifically and said the
frontend didn't need them.

What frontend verification *has* happened instead:
- `npm run lint` (oxlint) — clean except two pre-existing warnings unrelated to this work
  (an unused import in `DashboardPage.jsx`, a `useEffect` dependency in `TranscriptEditor.jsx`).
- Manual verification: both dev servers started and smoke-tested together (backend +
  frontend + Postgres), page-layout CSS fix confirmed to compile and hot-reload cleanly.

**This is the biggest gap in the testing story.** If frontend testing is wanted, see Next
Steps below.

---

## 4. Model (ASR) evaluation

Separate from the pytest suite — this measures the actual Whisper fine-tune's transcription
quality, not application code. Lives in `final-scripts/` (evaluation/training scripts) and
`ErrorAnalysis/` (results).

### 4a. Headline WER/CER across fine-tuning runs

From `final-scripts/finetune_tracker.csv`:

| Run | Type | Val WER / CER % | Test WER / CER % | English-forgetting delta | Status |
|---|---|---|---|---|---|
| run1 (lr3e-5, bs32) | full | 19.96 / 4.17 | 17.08 / 3.49 | +76.59 pts — **severe** | done |
| run6 (lr3e-5, bs32, AMD) | lora | 23.81 / 6.46 | 21.01 / 5.73 | +69.47 pts — **severe** | done |
| run2 (lr3e-5, bs32, lora) | lora | 61.63 / 18.81 | 59.07 / 18.03 | not measured | interrupted |
| run3 (lr1e-4, r32, lora) | lora | 26.01 / 7.26 | 25.99 / 7.06 | +2.76 pts — mild | done |
| full-lr1e-5-e4-cosine-bs64 | full | 28.38 / 8.03 | 28.47 / 7.93 | +1.79 pts — mild | done |
| **run5** (v4, lr3e-5, bs64, resumed) | full | 22.41 / 6.65 | **19.06 / 4.90** | not measured | done |

**Run5 is the current best test-set result** (19.06% WER / 4.90% CER on 15,860 held-out
samples) and is the run behind the currently-deployed model and the voice-command Whisper
checkpoint.

### 4b. Error-analysis clustering (`ErrorAnalysis/<run>/error_analysis/`)

For each run's predictions, `error_analysis.py` buckets every wrong sample by severity and
groups failures into 8 TF-IDF/KMeans clusters (character n-grams, so it works on Sinhala
script without a tokenizer) — turning "the model is 20% wrong" into "the model is wrong in
these specific, nameable ways":

- **Word-boundary/compounding errors** — the single largest failure cluster in nearly every
  run (e.g. `තුන්වැදෑරුම්ය` → `තුන් වැදෑරුම් ය`). Traced to inconsistent spacing conventions
  across the source datasets, not a model-capacity problem.
- **Conjunct-consonant/ZWJ errors** — Sinhala's zero-width-joiner conjuncts, inconsistently
  encoded in training data.
- **Colloquial vs. formal register** — the single largest driver of minor/moderate errors
  across every run (`කියල` ⇄ `කියලා`).
- **Degenerate repetition ("hallucination loop")** — present in the LoRA runs' worst
  individual samples, absent from the full fine-tune.

Full narrative and per-run numbers: `ErrorAnalysis/Analysis.md`.

### 4c. Voice-command embedding validation

Separate from transcription-quality testing — this validates whether the *voice fingerprint*
matching (used alongside fuzzy text matching for commands like "delete") actually
discriminates one spoken command from another. From this session's regenerated analysis
(`command_embedding_similarities_en.csv`, 435 pairwise comparisons across 30 real clips, 6
commands):

| Technique | Cohen's d (separation) | Overlap between same/different-command pairs |
|---|---|---|
| Euclidean distance | 4.996 | 0.8% |
| Manhattan distance (what the app uses) | 4.891 | 0.8% |
| Cosine similarity | 3.857 | 0.8% |
| Pearson correlation | 3.856 | 0.8% |

All four techniques separate cleanly; Manhattan (the one the app actually runs on) is a close
second to Euclidean and was kept rather than switched, since the gap is small and switching
would mean re-tuning every threshold in the matching pipeline for a marginal gain.

---

## 5. What was found, fixed, and re-tested this session

A real bug was found through this exact testing loop — not from the pytest suite (which was
all green throughout), but from **manually testing recorded commands and reading the backend
logs**, then confirmed numerically:

- **Symptom:** commands like "eka", "deka", and "next" weren't being recognized reliably in
  COMMAND mode.
- **Root cause 1:** the wake word ("zimi") had been baked directly into every command phrase
  (e.g. `"zimi එක"`). Fuzzy-match scoring is length-normalized, so a short command lost most
  of its discriminating power whenever Whisper failed to transcribe "zimi" — which, measured
  from real logs, happened roughly 62% of the time. Worse, the shared "zimi" prefix inflated
  similarity *between* different short commands (`eka` vs `deka` scored 83% instead of the
  correct 50%), risking cross-command confusion even when transcription worked.
  - **Fix:** wake-word detection was separated from command matching entirely. "Zimi" is now
    detected independently (by voice fingerprint first, with a text fallback for spellings
    `zimi`/`zini`/`සිමි`), arming a 3-second window in which exactly one bare command phrase
    is matched — restoring `eka`/`deka` to their natural, well-separated similarity.
  - **Re-tested:** `test_commands.py` now asserts the wake word is never part of a matchable
    phrase and that `eka`/`deka` score below the match threshold against each other;
    `test_streaming_commands_route.py` gained 15+ new tests covering the arm/expire/one-shot
    behavior, both wake spellings, voice-only wake detection, and — critically — that the
    "zimi" utterance itself is never saved into a student's note.
- **Root cause 2:** the VAD silence cutoff for COMMAND mode (300ms) was tuned for
  single-word commands and was cutting "zimi [pause] makanna" into two separate clips before
  the phrase-length change made that pause-tolerance necessary. Diagnosed but the config
  value itself was not the fix path taken — the wake/command split above absorbs this by
  design (each half is independently useful now, rather than needing to survive as one clip).
- **Also found and fixed:** 5 orphaned embedding samples for a `wake` command id that no
  longer existed in the matching vocabulary were still being loaded and matched (up to 0.878
  similarity), silently sending an unhandled `wake` command to the frontend. Cleaned up as
  part of re-enrollment.

This is the kind of bug automated unit tests alone would *not* have caught, because every
individual function was behaving correctly — the bug was in how real, noisy Whisper output
interacted with the matching design. It was found by generating real similarity numbers from
real recordings and logs, which is why §4c's embedding-technique comparison exists as a
standing tool, not a one-off.

### And one the tests themselves got wrong — caught by CI on its first run

Two of the new wake-gate tests passed locally but **failed the moment CI ran them**:

```
FAILED test_wake_detected_by_voice_when_whisper_mangles_the_word
       assert [] == [{'type': 'armed', 'seconds': 3.0}]
FAILED test_session_keeps_wake_samples_out_of_the_command_bank
       assert {} == {'next': ['n']}
```

**Root cause:** `voice_command_embedding_matching_enabled` defaults to `False` in
`config.py`, but the local `.env` sets it to `true` — and pydantic-settings reads `.env`
during tests. So both tests were passing for the wrong reason: they depended on ambient
developer configuration rather than declaring what they needed. With the flag off, the code
under test never loads the bank (`streaming.py:204`) and never computes an embedding
(`streaming.py:500`), so neither behaviour could occur.

`.env` is gitignored because it holds the database password, so CI — correctly — had none.

**Fix:** each test now sets the flag explicitly via `monkeypatch`, which the pre-existing
tests in that file already did. Verified three ways: locally, with the flag forced off, and
against CI's exact package versions in a fresh virtualenv.

This is worth recording because it's the canonical argument for CI: **a test that passes only
because of your local environment isn't testing what it claims to.** It would have stayed
green on one machine indefinitely while failing for every other contributor. The very first
CI run found it.

---

## 6. Next steps

Roughly in priority order:

1. **Pin dependency versions.** `backend/requirements.txt` has **zero version constraints**,
   so the same commit installs different libraries in each environment — currently
   `torch 2.13.0` locally versus `2.9.1` on the server and in CI, and `transformers 5.15.1`
   versus `5.17.0`. It happens to be harmless today (verified: the full suite passes on the
   newest versions), but it means builds aren't reproducible and a future release can break
   the deployment with no code change. `pip freeze > requirements.txt` or a lockfile.
2. **Add a frontend test suite.** No framework is installed at all right now. Vitest + React
   Testing Library would be the natural fit given this is a Vite project — start with the
   highest-value, highest-risk surfaces: the voice-command WebSocket hook
   (`useVoiceCommands.js`, since that's exactly where the bug in §5 lived on the client side),
   and the quiz/MCQ answer flow.
3. **Extend the eka/deka/tuna/hathara/cancel/answer/zimi voice-command clips into
   `storage/voice_samples/`** so `command_embedding_similarities_en.csv` and the cluster plot
   cover the *entire* command vocabulary, not just the original 6 (next/previous/delete/
   submit/save/stop). Right now the newer commands are enrolled in the live database but
   absent from this standing analysis.
4. **Re-run the English catastrophic-forgetting check on run5**, the current best/deployed
   model — it's the one run in the tracker missing that measurement. Given run3 and the
   other full-fine-tune run both showed only "mild" forgetting, run5 is likely fine, but it's
   an unverified gap, not a confirmed pass.
5. **Address the word-boundary/compounding and conjunct-consonant error clusters** identified
   in §4b — both are described in `Analysis.md` as *data*-consistency problems (inconsistent
   spacing/ZWJ conventions across the four source corpora), not model-capacity problems,
   meaning a normalization pass over the training transcripts is likely to help more than
   further training on the same data.
6. **Make local test runs hermetic too.** CI is now isolated (disposable Postgres, no
   `.env`), but running `pytest` locally still reads the dev database *and* the developer's
   `.env` — which is exactly what hid the bug in §5. A `conftest.py` that ignores `.env` and
   points at a throwaway database would make local runs match CI, so surprises surface
   before pushing rather than after.
7. **Consider a pre-push hook.** CI catches these, but only after you've pushed. Running the
   suite locally on `git push` would shorten the feedback loop.

**Done since the last revision:** CI wiring (§1b) and deployment smoke tests (§2) — both
previously listed here as outstanding.
