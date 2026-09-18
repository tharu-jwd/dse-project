# Testing Report

Last updated 2026-09-17. Covers the three layers this project actually has testing for:
**backend**, **frontend**, and **model (ASR) evaluation**. Frontend has no automated test
suite yet — that's called out explicitly rather than skipped over.

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
| `test_commands.py` | Fuzzy text matching (`skeleton()`, `match_command()`) against the Sinhala/English command vocabulary; the wake word is correctly excluded from matchable phrases; `eka`/`deka` stay cleanly separated (< 80% similarity) now that phrases aren't prefixed. | Direct regression guard for the exact bug reported during manual testing (§4) — commands scoring too low when transcribed correctly, or too high against the wrong command. |
| `test_embeddings.py` | The vector math underneath voice-fingerprint matching: pooling, L2 normalization, cosine/Manhattan similarity, `best_match()` — including zero-vector and empty-bank edge cases. | These are the functions a silent numeric bug (e.g. a NaN from dividing by zero on a silent clip) would hide in; each edge case here was a real, plausible failure mode, not a hypothetical. |
| `test_command_resolution.py` | The decision table that combines the fuzzy-text and embedding channels: execute / confirm / ignore, and the higher bar required for destructive commands (`delete`, `submit`). | This is the safety-critical logic — it's what stops an ambiguous or accidental utterance from deleting a student's work. |
| `test_inference_kwargs.py` | Whisper is only biased toward command words in COMMAND mode, never in NOTE/dictation mode. | Prevents ordinary dictation from being silently nudged toward command vocabulary. |
| `test_streaming_buffer.py` | The rolling audio buffer: correct timestamps across VAD-triggered finalizes and forced cuts, overlap handling at the boundaries. | A timestamp bug here would silently corrupt every transcript segment's start/end time — hard to spot by eye, easy to catch here. |
| `test_streaming_commands_route.py` | The WebSocket route end-to-end (with a fake socket, no real network): debouncing repeated commands, the "listening" UI signal, and — as of this session — the full wake-word gate (arm/expire/one-shot-unlock, both wake-word spellings, voice-only detection, "zimi" never saved as note text). | This is where the actual bug from manual testing was fixed and is now regression-tested: a command without a preceding wake word is provably ignored, not just "seems to work." |
| `test_voice_enrollment.py` | Enrollment: samples accepted/rejected by similarity, per-language isolation, the wake word can be enrolled and loads into the matching bank correctly. | Runs against a real (dev) Postgres database — the one file in the suite that isn't fully isolated (see §5). |

### Known limitation in the backend suite

`test_voice_enrollment.py` talks to the real development database rather than an isolated
test database (documented in `test/backend/conftest.py`). Each test cleans up its own
throwaway user, but this means the suite isn't fully hermetic and can't run without Postgres
up. Flagged, not yet fixed.

---

## 2. Frontend — no automated tests

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

## 3. Model (ASR) evaluation

Separate from the pytest suite — this measures the actual Whisper fine-tune's transcription
quality, not application code. Lives in `final-scripts/` (evaluation/training scripts) and
`ErrorAnalysis/` (results).

### 3a. Headline WER/CER across fine-tuning runs

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

### 3b. Error-analysis clustering (`ErrorAnalysis/<run>/error_analysis/`)

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

### 3c. Voice-command embedding validation

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

## 4. What was found, fixed, and re-tested this session

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
real recordings and logs, which is why §3c's embedding-technique comparison exists as a
standing tool, not a one-off.

---

## 5. Next steps

Roughly in priority order:

1. **Isolate `test_voice_enrollment.py` from the shared dev database.** Either a
   Dockerized/testcontainers Postgres fixture, or an in-memory SQLite schema sufficient to
   satisfy the foreign-key constraint. Currently the only non-hermetic part of the backend
   suite and the only one that can't run without Postgres up.
2. **Add a frontend test suite.** No framework is installed at all right now. Vitest + React
   Testing Library would be the natural fit given this is a Vite project — start with the
   highest-value, highest-risk surfaces: the voice-command WebSocket hook
   (`useVoiceCommands.js`, since that's exactly where the bug in §4 lived on the client side),
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
   in §3b — both are described in `Analysis.md` as *data*-consistency problems (inconsistent
   spacing/ZWJ conventions across the four source corpora), not model-capacity problems,
   meaning a normalization pass over the training transcripts is likely to help more than
   further training on the same data.
6. **CI wiring.** None of this — backend pytest, lint, or model evaluation — currently runs
   automatically on push/PR as far as this repo shows. Once the database-isolation item (1)
   is done, the backend suite is fast and hermetic enough to gate merges.
