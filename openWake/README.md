# openWake — English command classifier

Trains and evaluates the model that recognizes English voice commands
(`next`, `previous`, `save`, `submit`, `delete`, `cancel`, `stop`, `option1`-
`option4`) straight from raw audio — no transcript involved. This is what
`backend/app/streaming/audio_command_classifier.py` loads at runtime; see
"How it's used in the app" below for the full path from mic to command.

Sinhala command mode is unaffected by any of this — it stays on the older
fuzzy-transcript + speaker-embedding path (`command_resolution.resolve_command`),
untouched. This classifier is English-only.

## Pipeline

```
data_train/ (13 labels, incl. "none")
      │  generate.py  - TTS-generated clips + augmentation
      ▼
data_split/{train,val}/     split_dataset.py  - stratified split
      │
      ▼
features/{train,val}.npz    extract_features.py  - openWakeWord's EAR
      │                     (AudioFeatures) embeds each clip to (16, 96)
      ▼
train_classifier.py  - fits a scikit-learn pipeline (StandardScaler +
                        classifier) on the embeddings, saves
                        results/<Model>/model.joblib + metrics
      │
      ▼
test_my_recordings.py  - held-out real-voice sanity check
                        (my_recordings/, never trained on)
```

`prepare_audio.py` is shared between training (`extract_features.py`) and
runtime (`audio_command_classifier.py`) so a clip is prepared identically
both times: trim silence, then pad/crop to 3.0s, centered.

## Model comparison

Three classifiers were trained and evaluated on the same train/val
embeddings (6,306 train / 1,110 val clips, 13 labels): LogisticRegression,
RandomForest, and MLP. Full outputs per model are in `results/<ModelName>/`
(`metrics.json`, `classification_report.txt`, `confusion_matrix.csv`,
`confusion_matrix.png`).

**Data split clarification:** `my_recordings/` (your own real voice, 10 clips
per label) is **not** used to train any of these models — `generate.py`'s
`generate_my_voice()` step was skipped for this run, so none of `data_train/`,
and therefore none of `data_split/train` or `data_split/val`, contains your
voice or augmented variants of it. Instead, `my_recordings/` is reserved
entirely as a held-out real-voice test set, run through the trained models by
`test_my_recordings.py`. This keeps the "held-out test" results below honest —
no leakage between what the models were trained/validated on (TTS-generated
clips) and what they're tested on (your actual voice).

### Headline metrics

| Model | Accuracy | Macro F1 | Weighted F1 | False-accepts from "none" |
|---|---|---|---|---|
| **MLP** | **97.7%** | **0.981** | **0.978** | **14/234** |
| LogisticRegression | 97.2% | 0.977 | 0.972 | 19/234 |
| RandomForest | 96.1% | 0.967 | 0.962 | 28/234 |

### Per-class F1 (lowest LogisticRegression scores first)

| Class | LogisticRegression | RandomForest | MLP |
|---|---|---|---|
| option4 | 0.90 | 0.90 | 0.92 |
| none | 0.95 | 0.93 | 0.96 |
| option2 | 0.95 | 0.91 | 0.95 |
| stop | 0.98 | 0.95 | 0.97 |
| submit | 0.98 | 0.97 | 0.99 |
| option1 | 0.98 | 0.99 | 0.99 |
| next | 0.99 | 0.98 | 0.99 |
| save | 0.99 | 0.98 | 0.99 |
| start | 0.99 | 1.00 | 0.99 |
| previous | 0.99 | 0.99 | 0.99 |
| cancel | 0.99 | 0.98 | 1.00 |
| delete | 0.99 | 0.98 | 1.00 |
| option3 | 1.00 | 0.99 | 1.00 |

### Held-out test: real voice (`my_recordings/`, never trained on)

120 clips (10 per label, 12 labels — `my_recordings/` has no "none" class),
run through the already-trained models via `test_my_recordings.py`. Full
outputs per model are in `results/<ModelName>/held_out_test/`.

| Model | Accuracy | Macro F1 | Val accuracy (TTS, for comparison) |
|---|---|---|---|
| **LogisticRegression** | **99.2%** | **0.992** | 97.2% |
| **MLP** | **99.2%** | **0.992** | 97.7% |
| RandomForest | 98.3% | 0.987 | 96.1% |

Per-class misses (all three models): 1/10 `save` clips misread as `option2`
(likely near "option2" acoustically), plus RandomForest also missed 1/10
`delete` clips. Every other class is 10/10 across all three models. This
matches production: a live "next" clip that clipped at full volume was
misread as `previous`/`delete` during testing — clean input level, not the
model, is the main failure mode observed so far.

**Why real-voice accuracy is higher than the TTS val accuracy:** the val set
carries the full 13-label difficulty (including `none`, the hardest class,
and `option2`/`option4` which are the weakest classes on TTS data) at a much
larger, more varied scale — 1,110 clips across many TTS voices/accents vs.
120 clips of one consistent voice with no `none` class. This isn't a red flag;
it's a smaller, easier, but leakage-free sanity check, not a replacement for
the full validation numbers above.

### Analysis

- **MLP wins on every headline metric** — highest accuracy, highest macro-F1,
  and fewest false-accepts from "none" (14/234 vs 19 for LogisticRegression
  and 28 for RandomForest). It has no `class_weight` support, yet still
  handled the imbalance fine — the minority classes (384-540 clips) are
  large enough for a shallow net on top of strong pre-trained embeddings.
- **RandomForest is the weakest of the three** on every metric, worst on
  "none" recall (88% vs 92-94%) — the model most likely to false-trigger a
  command mid-session. It is not shipped anywhere (see below).
- **option2 and option4 are the hard classes for all three models**
  (lowest precision, 0.84-0.91). This is a data/separability issue, not a
  model-choice issue — switching classifiers won't fix it; worth listening
  to those clips to check if they're acoustically close to "none" or to
  each other.
- **Gap between MLP and the LogisticRegression baseline is small**
  (~0.5pt accuracy, ~0.4pt macro-F1). LogisticRegression is far cheaper to
  train/run and supports `class_weight` natively, so it would be a
  reasonable fallback if inference latency/footprint on-device is ever a
  hard constraint.
- **Decision: MLP is what ships**, specifically for the lower "none"
  false-accept rate — that's the costliest error type in an active session
  (fires an unwanted action). Its `model.joblib` is copied into
  `backend/app/models/command_classifier/model.joblib` and is what
  `audio_command_classifier.py` loads at runtime; see
  `backend/app/core/config.py`'s `voice_command_audio_model_path`.
- **The held-out real-voice test confirms this ranking holds**: LogisticRegression
  and MLP tie at 99.2% on real voice, RandomForest trails at 98.3% —
  consistent with RandomForest being the weakest on TTS val data too. Since
  `my_recordings/` has no "none" clips, it can't validate the false-accept
  concern above; that still needs checking with real "none"/silence
  recordings, not just the TTS-generated "none" variety.

## How it's used in the app

English COMMAND-mode streaming sessions only (`backend/app/api/routes/streaming.py`).
Sinhala sessions never call any of this.

1. **Wake word ("zimi"), by speaker embedding.** Exactly the same mechanism
   Sinhala uses (`app.streaming.embeddings.best_match` against the
   student's enrolled samples) — English has no separate wake model. A
   student who already enrolled "zimi" once (Sinhala) doesn't need to
   enroll it again: an English session with no English wake samples falls
   back to loading the Sinhala ones, since "zimi" is the same word either
   way (see `_run_session` in `streaming.py`).
2. **Wake detected → the wake window opens** (`armed_until`), same as
   Sinhala's flow.
3. **Next clip in the window → classified from its raw audio**, not its
   Whisper transcript, by this classifier
   (`app.streaming.audio_command_classifier.classify`):
   - `prepare_audio.py`'s trim/pad-to-3s logic runs on the clip.
   - openWakeWord's `AudioFeatures` (EAR) embeds it, same as training.
   - The MLP pipeline's `predict_proba` picks the top label.
   - Below `voice_command_audio_confidence_threshold` (default 0.8), or a
     `none` prediction, resolves to no command — same "don't guess" rule
     Sinhala's fuzzy/embedding combination follows.
   - A recognized label maps to the app's command ids (`option1` →
     `option_1`, etc. — see `_LABEL_TO_COMMAND_ID` in
     `audio_command_classifier.py`).
4. **One wake word unlocks exactly one command** — executing one clears
   `armed_until`, same rule as Sinhala.

`app.py` in this directory is an earlier, standalone prototype of this same
idea (a continuously-listening `hey_nexa`/`bye_nexa` session with no backend
integration) and is **not** used by the real app — the backend never imports
it. It's kept only as a reference for the classifier + EAR call pattern.
`command_model.joblib`, `models/hey_nexa.onnx`, and `models/bye_nexa.onnx`
belong to that prototype only.

## What's gitignored, and why

Large or personal artifacts that are either regeneratable from the scripts
above, or superseded, are gitignored rather than committed:

- **`my_recordings/`** — your own held-out voice samples. Personal audio
  data; regenerate your own by recording into this folder (see
  `test_my_recordings.py`'s docstring for the expected layout).
- **`data_split/`, `data_train/`** — regenerated by `generate.py` +
  `split_dataset.py`; too large and fully reproducible to commit.
- **Unused classifiers** — `results/RandomForestClassifier/model.joblib`
  and `results/LogisticRegression/model.joblib`. Neither ships anywhere
  (see "Decision: MLP is what ships" above); their metrics/reports/confusion
  matrices stay committed since those are what this README's comparison is
  built from, and are small. `results/MLPClassifier/model.joblib` (the one
  that does ship) stays committed too, as the source of truth for the copy
  under `backend/app/models/command_classifier/`.
- **The standalone prototype's models** — `models/hey_nexa.onnx`,
  `models/bye_nexa.onnx`, and `command_model.joblib`. Only `app.py` (unused
  by the backend) reads these; see "How it's used in the app" above.

Re-run the relevant script to regenerate anything gitignored here.
