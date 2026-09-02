# Voice command enrollment

Two independent ways the app recognizes a spoken command during live
note-taking ("delete", "stop", ...):

1. **Fuzzy text matching** (`backend/app/streaming/commands.py`) — Whisper
   transcribes the utterance, the text is fuzzy-matched against the known
   command phrases. Always on, no setup required.
2. **Speaker-embedding matching** (`backend/app/streaming/embeddings.py`) —
   the audio itself (not the transcribed text) is compared against a bank
   of the student's own recordings of each command. Off by default, and
   only does anything for a student who has enrolled.

Enrollment is what fills that bank. It's optional — voice commands work
from the fuzzy path alone with no enrollment at all — but it catches
cases the text path can't, because it doesn't depend on Whisper
transcribing the phrase correctly.

## Why this exists

Short command phrases sometimes transcribe unreliably, and some
students' speech is transcribed inconsistently regardless of model
quality. Enrollment lets the app recognize a command by *how the
student says it*, independent of whether Whisper gets the words right.

Concretely, on a real held-out recording from this project: Whisper
transcribed a "delete" (මකන්න) recording as "මක් කන්නේ" — the fuzzy path
correctly found no match. The embedding path, comparing the same audio
against four other enrolled "delete" takes, recognized it anyway
(85.4% similarity on the metric described below, comfortably above the
threshold). That's the gap this feature closes.

**Similarity metric**: `best_match` scores on a rescaled Manhattan
distance, not cosine similarity — a same-recording-session comparison
across all six commands found Manhattan distance separates
same-command from different-command pairs slightly better (Cohen's d
2.47 vs 2.27; see `scripts/validate_command_embeddings.py --csv` for
the numbers, which also reports Euclidean and Pearson for comparison).
The rescale (`embeddings.manhattan_similarity`) isn't an arbitrary
number — for L2-normalised vectors of dimension `dim`, Manhattan
distance is bounded above by `2·√dim`, so `1 - distance / (2·√dim)`
gives a principled 0–1-ish "higher is better" score without needing
re-deriving if the embedding dimension ever changes.

## How it works, end to end

### 1. Enrollment (recording samples)

Page: **Settings → Set up voice commands** (`/settings/voice-commands`,
students only — `frontend/src/pages/VoiceEnrollmentPage.jsx`).

For each of the six commands, the student records a short clip (reuses
the same record → `MediaRecorder` blob → upload pattern as
`AudioRecorder.jsx`, not a second recorder implementation). On upload:

1. The backend converts whatever format the browser produced (webm/opus,
   typically) to 16kHz mono via `ffmpeg`.
2. The audio is embedded (`app/streaming/embeddings.py`: VAD-trimmed,
   encoder-only Whisper forward pass, mean-pooled, L2-normalized).
3. If this command already has samples, the new embedding is compared
   against them. Below `voice_enrollment_min_sample_similarity`
   (default 0.80), it's **rejected** — a mis-recorded sample would
   otherwise poison the bank — and the student is asked to say it again.
   The first sample for a command is always accepted (nothing to compare
   against yet).
4. On acceptance, the embedding is stored in `command_enrollments`,
   keyed by `(user_id, command_id, sample_index)`.

Resumable by design: nothing requires finishing all six commands in one
sitting, and the page shows per-command progress
(`collected`/`required`) so a student can pick up where they left off.

### 2. Runtime matching (using the bank)

When a live note-taking session starts (`app/api/routes/streaming.py`),
the student's full bank is loaded once into memory — small (six
commands × a handful of samples × one 768-float vector each). Every
*finalized* utterance (never a mid-utterance "partial") is then checked
by `app/streaming/command_resolution.py`, which combines both signals:

| Fuzzy | Embedding | Result |
|---|---|---|
| strong | strong, same command | execute |
| strong | weak/no match | execute |
| weak/no match | strong | execute |
| both weak, but *some* borderline candidate | | ask for confirmation (kept as text, client notified) |
| strong, but the two **disagree** | | ask for confirmation, never guess |
| nothing resembling a command on either side | | ordinary dictation, silent |

Destructive commands (`submit`, `delete`) don't get an extra
confirmation step on top of this — they simply need a *higher* score to
count as "strong" within whichever signal is calling them (see
`voice_command_destructive_threshold` and
`voice_embedding_destructive_threshold`). Nothing is ever executed on a
guess: the combination table above is implemented literally in
`resolve_command`, and "confirm" always leaves the spoken words in the
note rather than discarding or acting on them.

A student with no enrollment bank (or with
`voice_command_embedding_matching_enabled` off) gets exactly the
fuzzy-only behaviour that existed before this feature — enrollment is
additive, never a prerequisite.

### 3. What each command actually does (client-side)

`resolve_command` only ever decides *which command id* was spoken, with
what confidence — it has no idea what that id should do on whatever
page the student is on. The backend's COMMAND-mode WebSocket
(`app/api/routes/streaming.py`) just forwards an `outcome: "execute"`
decision as `{"type": "command", "command": "<id>"}` with zero
server-side effect; every page listening via `useVoiceCommands`
(`frontend/src/hooks/useVoiceCommands.js`) maps the six ids to its own
actions:

| Command | Self-study note (live, mid-dictation) | Transcript Review page | Quiz answer page |
|---|---|---|---|
| `delete` | deletes the last line, server-side | — | — |
| `stop` | ends the recording session | — | — |
| `save` | — | saves the draft (no-ops with "Nothing to save" if nothing changed) | — |
| `next` | — | — | advances a question (only if the current one is answered) |
| `previous` | — | — | goes back a question |
| `submit` | — | **opens the finalize confirmation dialog** — never finalizes by itself | **opens the submit confirmation dialog** — only once every required question is answered; never submits by itself |

`delete`/`stop` are the only two commands with a real server-side effect,
and only inside a live NOTE session (see `_ACTIONABLE_NOTE_COMMANDS` in
`streaming.py`) — everywhere else, including `delete`/`stop` themselves
outside of live note-taking, a recognized command is just handed to the
page, which decides what it means there.

**`submit` is deliberately never a one-step action anywhere.** Saying it
opens the same confirmation dialog (`ConfirmDialog`) the mouse-driven
"Finalize"/"Review & submit" button opens — the student still has to
confirm the dialog themselves, exactly like a destructive fuzzy/embedding
match still requires the higher `*_destructive_threshold` bar rather than
a normal one. This is intentional, not a missing feature: an
irreversible action (finalizing a transcript, submitting a quiz) should
never fire off a single misheard "submit" with no chance to back out.

## Re-running after a model change

Every stored embedding is stamped with `voice_embedding_model_version`
(currently `"whisper-sinhala1-ct2"`) at the time it was recorded.
`load_bank()` skips any row whose stamp doesn't match the *currently
configured* model — a stale embedding is silently useless (comparing
vectors from two different model spaces is meaningless), so it's
excluded rather than compared incorrectly. A warning is logged when this
happens.

If you retrain or reconvert the streaming checkpoint:

1. Bump `voice_embedding_model_version` in `app/core/config.py` (or the
   `.env` override) to a new value.
2. Existing enrollments are now automatically excluded from matching —
   students effectively fall back to fuzzy-only until they re-enroll.
   There is currently no bulk re-embedding tool; the correct fix is
   re-recording (`DELETE /voice-enrollment/{command_id}` then submit new
   samples), since a changed encoder can shift what "sounds similar"
   even for the same speaker.
3. Re-run the validation harness on fresh recordings before trusting the
   new checkpoint's numbers:

   ```bash
   python -m scripts.validate_command_embeddings path/to/wav_dir
   ```

   Update `voice_embedding_similarity_threshold` /
   `voice_embedding_destructive_threshold` from its suggested threshold
   output, not by guessing. The current values (0.828 / 0.85, under
   Manhattan similarity) came from a real 31-clip recording session this
   way — see the script's
   docstring for what "good" output looks like.

## Interpreting the logs

Two log lines matter, both at `INFO`:

**`voice_command_attempt`** (from `commands.py`, the fuzzy path alone —
fires on every finalized utterance regardless of enrollment):

```
voice_command_attempt transcript='මක් කන්නේ' matched=None score=42.0 avg_logprob=-0.177
```

**`voice_command_decision`** (from `command_resolution.py`, only when
embedding matching is enabled and the student has a bank — the combined
decision):

```
voice_command_decision transcript='මක් කන්නේ' outcome=execute command=delete
fuzzy=None(None) embedding=delete(0.854) agreed=False
```

To find out which phrases are unreliable in practice: grep for
`voice_command_attempt` over a period of real use and look at how often
`matched=None` shows up for utterances a student actually intended as a
command (you won't know intent from the log alone — cross-reference
against `command_maybe`/`command` WebSocket messages the frontend
received, or just ask the student). A command that's frequently
`matched=None` on the fuzzy side but `outcome=execute` via embedding in
`voice_command_decision` is direct evidence the embedding path is
pulling its weight for that phrase, same as the delete example above. A
command that's failing on *both* signals consistently is a candidate to
reword, or to double down on enrollment prompts for.

## A/B testing

`voice_command_embedding_matching_enabled` (default `False`) is a single
global flag. To compare fuzzy-only vs. combined:

1. Run with it `False` for a period, collect `voice_command_attempt`
   logs.
2. Flip it `True`, collect `voice_command_decision` logs for the same
   students (who need to have enrolled in between).
3. Compare `outcome` distributions and, where you can get ground truth
   (ask the student, or note when they immediately repeat themselves —
   a sign the previous attempt didn't register), false-negative rates
   between the two periods.

There's no per-user override yet — it's an app-wide setting, so
"A/B" here means "before/after," not two simultaneous cohorts. Splitting
by user would need a small addition (e.g. a settings check keyed on
user id) if you want a true concurrent A/B test.
