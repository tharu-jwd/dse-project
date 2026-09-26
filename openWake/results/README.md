# Command classifier — model comparison

LogisticRegression vs RandomForest vs MLP, trained via `train_classifier.py` on
the same train/val embeddings (6,306 train / 1,110 val clips, 13 labels).
Full outputs per model are in `results/<ModelName>/` (`metrics.json`,
`classification_report.txt`, `confusion_matrix.csv`, `confusion_matrix.png`).

**Data split clarification:** `my_recordings/` (your own real voice, 10 clips
per label) is **not** used to train any of these models — `generate.py`'s
`generate_my_voice()` step was skipped for this run, so none of `data_train/`,
and therefore none of `data_split/train` or `data_split/val`, contains your
voice or augmented variants of it. Instead, `my_recordings/` is reserved
entirely as a held-out real-voice test set, run through the trained models by
`test_my_recordings.py`. This keeps the "held-out test" results below honest —
no leakage between what the models were trained/validated on (TTS-generated
clips) and what they're tested on (your actual voice).

## Headline metrics

| Model | Accuracy | Macro F1 | Weighted F1 | False-accepts from "none" |
|---|---|---|---|---|
| **MLP** | **97.7%** | **0.981** | **0.978** | **14/234** |
| LogisticRegression | 97.2% | 0.977 | 0.972 | 19/234 |
| RandomForest | 96.1% | 0.967 | 0.962 | 28/234 |

## Per-class F1 (lowest LogisticRegression scores first)

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

## Held-out test: your real voice (`my_recordings/`, never trained on)

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
`delete` clips. Every other class is 10/10 across all three models.

**Why real-voice accuracy is higher than the TTS val accuracy:** the val set
carries the full 13-label difficulty (including `none`, the hardest class,
and `option2`/`option4` which are the weakest classes on TTS data) at a much
larger, more varied scale — 1,110 clips across many TTS voices/accents vs.
120 clips of one consistent voice with no `none` class. This isn't a red flag;
it's a smaller, easier, but leakage-free sanity check, not a replacement for
the full validation numbers above.

## Analysis

- **MLP wins on every headline metric** — highest accuracy, highest macro-F1,
  and fewest false-accepts from "none" (14/234 vs 19 for LogisticRegression
  and 28 for RandomForest). It has no `class_weight` support, yet still
  handled the imbalance fine — the minority classes (384-540 clips) are
  large enough for a shallow net on top of strong pre-trained embeddings.
- **RandomForest is the weakest of the three** on every metric, worst on
  "none" recall (88% vs 92-94%) — the model most likely to false-trigger a
  command mid-session.
- **option2 and option4 are the hard classes for all three models**
  (lowest precision, 0.84-0.91). This is a data/separability issue, not a
  model-choice issue — switching classifiers won't fix it; worth listening
  to those clips to check if they're acoustically close to "none" or to
  each other.
- **Gap between MLP and the LogisticRegression baseline is small**
  (~0.5pt accuracy, ~0.4pt macro-F1). LogisticRegression is far cheaper to
  train/run and supports `class_weight` natively, so it's a reasonable
  production default unless the false-accept reduction from MLP (19→14)
  matters enough to justify the extra complexity.
- **Recommendation: MLP**, specifically for the lower "none" false-accept
  rate — that's the costliest error type in the ACTIVE session (fires an
  unwanted action). Fall back to LogisticRegression if inference
  latency/footprint on-device is a hard constraint.
- **The held-out real-voice test confirms this ranking holds**: LogisticRegression
  and MLP tie at 99.2% on your actual voice, RandomForest trails at 98.3% —
  consistent with RandomForest being the weakest on TTS val data too. Since
  `my_recordings/` has no "none" clips, it can't validate the false-accept
  concern above; that still needs to be checked with real "none"/silence
  recordings of your own voice, not just the 13 TTS-generated "none" variety.
