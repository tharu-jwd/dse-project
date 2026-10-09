# Command Mode

Command mode lets a student control parts of the SinhaSpeech app **by voice**,
without a keyboard or mouse. It is an accessibility pathway for students with
writing or motor difficulties: a short spoken word ("next", "save", "delete",
an MCQ option number, …) is turned into an in-app action.

Command mode is **not** full voice control of the whole application — it covers
a fixed vocabulary of navigation, save/submit, MCQ-answer and
confirm/cancel commands in the flows that wire it in (transcript review and
quiz answering). Normal dictation (free speech → text) is a **separate** mode
and never touches the command vocabulary.

Related docs: [`voice-enrollment.md`](voice-enrollment.md) (how a student records
their own command samples), [`../openWake/README.md`](../openWake/README.md)
(the wake-word / English audio classifier training).

---

## 1. How it works end to end

```
Browser mic ──▶ WebSocket /ws (mode = COMMAND)
                     │
                     ▼
         StreamingTranscriber (faster-whisper, int8 CT2)
                     │   buffers audio, VAD splits on short silence
                     ▼
          Wake word?  ── voice fingerprint first, ASR text fallback
                     │   (arms a short command window)
                     ▼
        Finalized command utterance
                     │
                     ▼
   command_resolution.resolve_command(...)  ─────────────┐
     • embedding match  (embeddings.best_match)          │
     • (fuzzy text match — currently disabled)           │
     • English audio classifier (audio_command_classifier)│
                     │                                    │
                     ▼                                    │
        Outcome: "execute" | "confirm" | "none" ◀─────────┘
                     │
                     ▼
            Frontend acts on the command
          (useVoiceCommands.js) or asks the
          student to confirm a destructive one
```

- **Transport:** a single live WebSocket at `/ws`
  ([`backend/app/api/routes/streaming.py`](../backend/app/api/routes/streaming.py)).
  The same socket serves two modes — `NOTE` (dictation) and `COMMAND`. Only
  COMMAND mode applies anything from the command vocabulary.
- **Inference:** the CTranslate2 / faster-whisper streaming model
  ([`streaming/inference.py`](../backend/app/streaming/inference.py)), int8, CPU
  by default, with voice-activity detection
  ([`streaming/vad.py`](../backend/app/streaming/vad.py)) to cut an utterance on
  a short pause (commands use a tighter 300 ms silence than dictation's 500 ms).
- **Wake word:** a command window is **armed** by the wake word `zimi` (text
  variants `zimi`, `zini`, `සිමි`), detected by voice fingerprint first and
  transcript text as a fallback. The wake word is deliberately **not** part of
  any command phrase (a shared prefix distorted the length-normalised fuzzy
  scores). Defined as `WAKE_WORD` in
  [`commands.py`](../backend/app/streaming/commands.py).

---

## 2. Command vocabulary

Defined in [`backend/app/streaming/commands.py`](../backend/app/streaming/commands.py).
Two fixed phrase sets exist; **which one is live is a per-user DB setting**
(`User.command_language`), not a code change. The wording lives in code; what is
data-driven is the active language and the student's enrolled samples.

| Command id | Sinhala phrase | English phrase | Destructive |
|---|---|---|---|
| `next`      | ඊළඟට            | next     | |
| `previous`  | ආපසු            | previous | |
| `stop`      | නවත්වන්න        | stop     | |
| `save`      | සුරකින්න        | save     | |
| `submit`    | ඉදිරිපත් කරන්න  | submit   | ✓ |
| `delete`    | මකන්න           | delete   | ✓ |
| `option_1`  | එක              | one      | |
| `option_2`  | දෙක             | two      | |
| `option_3`  | තුන             | three    | |
| `option_4`  | හතර             | four     | |
| `cancel`    | අවලංගු කරන්න     | cancel   | |
| `answer`    | පිළිතුර          | answer   | |

`destructive` commands (`submit`, `delete`) are gated behind a higher
confidence bar and resolve to **confirm** rather than **execute** unless the
match is very strong — a misfired delete must never happen silently.

---

## 3. How a spoken command is resolved

[`backend/app/streaming/command_resolution.py`](../backend/app/streaming/command_resolution.py)
combines the available signals into one decision and returns an outcome:

- **`execute`** — act immediately.
- **`confirm`** — ask the student to confirm (used for destructive commands and
  borderline matches).
- **`none`** — no command matched; treat as nothing / ordinary input.

Signals:

1. **Speaker embedding match** (primary) —
   [`streaming/embeddings.py`](../backend/app/streaming/embeddings.py) encodes
   the clip with the Whisper **encoder only** (feature extractor → encode →
   mean-pool over time → L2-normalise) and compares it to the student's enrolled
   command embeddings. Scoring uses **Manhattan-distance-derived similarity**,
   chosen over cosine because it separated same-command from different-command
   pairs more cleanly on real recordings (Cohen's d 2.47 vs 2.27).
2. **Fuzzy text match** — matching the ASR transcript against the phrase
   vocabulary with RapidFuzz. **Currently disabled** in resolution (commands are
   identified from embeddings only); the code and thresholds remain for
   re-enabling.
3. **English audio classifier** —
   [`streaming/audio_command_classifier.py`](../backend/app/streaming/audio_command_classifier.py),
   the openWake-trained sklearn pipeline over openWakeWord EAR embeddings, used
   for the English MCQ option words.

---

## 4. Configuration

All in [`backend/app/core/config.py`](../backend/app/core/config.py)
(`Settings`), overridable by environment variable.

| Setting | Default | Meaning |
|---|---|---|
| `streaming_enabled` | `False` | Master switch for the live/command WebSocket. |
| `streaming_max_sessions_per_user` | `3` | Concurrent live sessions per user. |
| `streaming_command_vad_silence_ms` | `300` | Silence that ends a command utterance (tighter than dictation's 500 ms). |
| `voice_command_embedding_matching_enabled` | `False` | Turn on the embedding matcher (off by default so it can be enabled per deploy). |
| `voice_embedding_similarity_threshold` | `0.828` | Min embedding similarity to accept a command (max-margin value from the validation data). |
| `voice_embedding_min_clip_seconds` | `0.3` | Shorter clips are ignored. |
| `voice_embedding_model_version` | `whisper-sinhala1-ct2` | Enrolled samples are keyed to this; a version change invalidates old samples. |
| `voice_command_hotwords_enabled` | `True` | Whisper hotword biasing toward the vocabulary. |
| `voice_command_fuzzy_threshold` | `80.0` | Fuzzy accept threshold (fuzzy path currently disabled). |
| `voice_command_fuzzy_exact_override` | `95.0` | Near-exact skeleton match override for short, distinctive phrases. |
| `voice_command_destructive_threshold` | `95.0` | Higher bar for `submit`/`delete` on the fuzzy side. |
| `voice_command_logprob_floor` | `-0.5` | Reject low-confidence ASR before matching. |
| `voice_enrollment_min_sample_similarity` | `0.77` | Looser bar for accepting an enrolled sample at record time. |

---

## 5. Enrollment (prerequisite)

Embedding matching is **per student**: each student records several takes of
each command phrase, which become their enrolled embeddings. See
[`voice-enrollment.md`](voice-enrollment.md). Samples are stored as
`{command_id}_{n}.wav` and keyed to `voice_embedding_model_version`; changing
the streaming model invalidates them and the student re-enrolls.

- Backend: [`app/services/voice_enrollment.py`](../backend/app/services/voice_enrollment.py),
  routes [`app/api/routes/voice_samples.py`](../backend/app/api/routes/voice_samples.py).
- Frontend: [`VoiceEnrollmentPage.jsx`](../frontend/src/pages/VoiceEnrollmentPage.jsx),
  [`VoiceSampleCollectorPage.jsx`](../frontend/src/pages/VoiceSampleCollectorPage.jsx).

---

## 6. Frontend

- Hook: [`frontend/src/hooks/useVoiceCommands.js`](../frontend/src/hooks/useVoiceCommands.js)
  — opens the WebSocket in command mode, receives resolved outcomes, and drives
  the UI (execute vs. confirm).
- UI feedback: [`VoiceMeter.jsx`](../frontend/src/components/VoiceMeter.jsx)
  (mic activity).
- A student opts into voice interaction in accessibility settings
  (`User.command_language` selects the Sinhala or English phrase set).

---

## 7. Testing

- `test/backend/test_command_resolution.py` — execute / confirm / none decision logic.
- `test/backend/test_command_safety.py` — destructive-command thresholds.
- `test/backend/test_voice_enrollment.py`, `test_api_voice_enrollment.py` —
  enrollment and bank loading (real model is monkeypatched; CI never loads it).
- `frontend/src/hooks/useVoiceCommands.test.jsx` — frontend command handling.

---

## 8. Known limitations

- Command mode covers a **fixed vocabulary** in the flows that wire it in; it is
  not whole-app voice control.
- Embedding matching is **off by default** (`voice_command_embedding_matching_enabled`)
  and requires prior enrollment per student.
- Live/streaming inference is **CPU-bound**; concurrent sessions degrade — see
  the performance notes in the main report and [`DEPLOYMENT.md`](DEPLOYMENT.md).
